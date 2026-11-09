"""Integration contract against a real PostgreSQL and the Rust API with a fake Basalt engine.

Run after cargo build with DATABASE_URL pointing to a disposable test database.
"""
import json
import concurrent.futures
import os
import subprocess
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeBasalt(BaseHTTPRequestHandler):
    seen = []
    failed_once = False
    def log_message(self, *_args):
        pass
    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        self.seen.append(req)
        if req["messages"][-1]["content"] == "fail once" and not FakeBasalt.failed_once:
            FakeBasalt.failed_once = True
            body=b'{"error":"transient simulated inference failure"}'
            self.send_response(503)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps({"choices":[{"message":{"role":"assistant","content":"reply-"+str(len(self.seen))}}],
                           "usage":{"prompt_tokens":100,"completion_tokens":5,
                                    "prompt_tokens_details":{"cached_tokens":25 if len(req["messages"])>1 else 0}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake=ThreadingHTTPServer(("127.0.0.1",8080),FakeBasalt)
        threading.Thread(target=cls.fake.serve_forever,daemon=True).start()
        cls.key="integration-test-thread-key-012345"
        env=dict(os.environ,BASALT_THREAD_API_KEY=cls.key,
                 BASALT_THREAD_OWNER_ID="00000000-0000-0000-0000-00000000a001",
                 BASALT_THREAD_WORKSPACE_ID="00000000-0000-0000-0000-00000000a002",
                 BASALT_UPSTREAM_URL="http://127.0.0.1:8080")
        cls.env=env
        cls.proc=subprocess.Popen(["crates/target/debug/basalt-thread-api"],env=env,
                                  stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        for _ in range(100):
            try:
                code,_=cls.call("GET","/v1/threads")
                if code in (404,405):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            if cls.proc.poll() is not None:
                raise RuntimeError("Rust thread sidecar exited before readiness: "+
                    cls.proc.stderr.read().decode(errors="replace"))
            raise RuntimeError("Rust thread sidecar did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try: cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired: cls.proc.kill()
        cls.fake.shutdown()
        cls.fake.server_close()

    @classmethod
    def call(cls,method,path,data=None):
        raw=None if data is None else json.dumps(data).encode()
        req=urllib.request.Request("http://127.0.0.1:8095"+path,data=raw,method=method,
              headers={"Authorization":"Bearer "+cls.key,"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=30) as res:
                return res.status,json.loads(res.read())
        except urllib.error.HTTPError as e:
            return e.code,json.loads(e.read() or b'{}')

    def test_create_resume_idempotency_and_conflict(self):
        seen_before=len(FakeBasalt.seen)
        code,t=self.call("POST","/v1/threads",{"model":"fake-model"})
        self.assertEqual(code,201,t)
        tid=t["id"]; branch=t["branch_id"]
        first={"branch_id":branch,"expected_version":0,"parent_message_id":None,
               "idempotency_key":"req-1","content":"hello","max_tokens":32}
        code,r=self.call("POST",f"/v1/threads/{tid}/turns",first)
        self.assertEqual(code,200,r)
        self.assertEqual(r["status"],"COMMITTED")
        head=r["message_id"]
        code,replay=self.call("POST",f"/v1/threads/{tid}/turns",first)
        self.assertEqual(code,200,replay)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["message_id"],head)
        code,_=self.call("POST",f"/v1/threads/{tid}/turns",{**first,"content":"tampered"})
        self.assertEqual(code,409)
        code,view=self.call("GET",f"/v1/threads/{tid}")
        self.assertEqual(view["version"],1)
        self.assertEqual(view["head_message_id"],head)
        second={**first,"idempotency_key":"req-2","expected_version":1,
                "parent_message_id":head,"content":"continue"}
        code,s=self.call("POST",f"/v1/threads/{tid}/turns",second)
        self.assertEqual(code,200,s)
        code,history=self.call("GET",f"/v1/threads/{tid}/messages")
        self.assertEqual(code,200,history)
        self.assertEqual([x["role"] for x in history["data"]],
                         ["user","assistant","user","assistant"])
        self.assertEqual([x["content"] for x in history["data"] if x["role"]=="user"],
                         ["hello","continue"])
        self.assertTrue(all(x["content"].startswith("reply-") for x in history["data"] if x["role"]=="assistant"))
        self.assertEqual(len(FakeBasalt.seen),seen_before+2)
        self.assertEqual(len(FakeBasalt.seen[-1]["messages"]),3)
        code,metrics=self.call("GET",f"/v1/threads/{tid}/metrics")
        self.assertEqual(code,200,metrics)
        self.assertEqual(len(metrics["data"]),2)
        self.assertEqual([v["reported_cached_tokens"] for v in metrics["data"]],[0,25])
        self.assertTrue(all(v["ttft_ms"] is None for v in metrics["data"]))
        code,_=self.call("POST",f"/v1/threads/{tid}/turns",
                         {**second,"idempotency_key":"stale-key"})
        self.assertEqual(code,409)

    def test_ten_concurrent_requests_commit_only_one_head(self):
        status,t=self.call("POST","/v1/threads",{"model":"fake-model"})
        self.assertEqual(status,201,t)
        tid=t["id"];branch=t["branch_id"]
        def request(n):
            return self.call("POST",f"/v1/threads/{tid}/turns",{
              "branch_id":branch,"expected_version":0,"parent_message_id":None,
              "idempotency_key":f"race-{n}","content":f"message {n}","max_tokens":16})
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            answers=list(pool.map(request,range(10)))
        codes=[status for status,_ in answers]
        self.assertEqual(codes.count(200),1,answers)
        self.assertEqual(codes.count(409),9,answers)
        status,t=self.call("GET",f"/v1/threads/{tid}")
        self.assertEqual(status,200)
        self.assertEqual(t["version"],1)
        status,history=self.call("GET",f"/v1/threads/{tid}/messages")
        self.assertEqual((status,len(history["data"])),(200,2))

    def test_restart_reconstructs_from_postgres(self):
        status,t=self.call("POST","/v1/threads",{"model":"fake-model"})
        self.assertEqual(status,201,t)
        tid=t["id"];branch=t["branch_id"]
        msg={"branch_id":branch,"expected_version":0,"parent_message_id":None,
             "idempotency_key":"reboot","content":"persist me","max_tokens":16}
        status,result=self.call("POST",f"/v1/threads/{tid}/turns",msg)
        self.assertEqual(status,200,result)
        self.proc.terminate()
        self.proc.wait(timeout=5)
        self.__class__.proc=subprocess.Popen(
             ["crates/target/debug/basalt-thread-api"],env=self.env,
             stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        for _ in range(100):
            if self.proc.poll() is not None:
                raise AssertionError("sidecar restart failed: "+self.proc.stderr.read().decode())
            try:
                status,view=self.call("GET",f"/v1/threads/{tid}")
                if status==200: break
            except OSError:
                time.sleep(0.1)
        else: self.fail("sidecar restart readiness timeout")
        self.assertEqual(view["head_message_id"],result["message_id"])
        status,messages=self.call("GET",f"/v1/threads/{tid}/messages")
        self.assertEqual([x["content"] for x in messages["data"] if x["role"]=="user"],["persist me"])

    def test_explicit_forks_keep_independent_heads_and_history(self):
        status,t=self.call("POST","/v1/threads",{"model":"fake-model"})
        self.assertEqual(status,201,t)
        tid=t["id"]
        first={"branch_id":t["branch_id"],"expected_version":0,
               "parent_message_id":None,"idempotency_key":"root","content":"root",
               "max_tokens":16}
        status,out=self.call("POST",f"/v1/threads/{tid}/turns",first)
        self.assertEqual(status,200,out)
        head=out["message_id"]
        status,fork=self.call("POST",f"/v1/threads/{tid}/branches",{"from_message_id":head})
        self.assertEqual(status,201,fork)
        bid=fork["id"]
        status,new=self.call("POST",f"/v1/threads/{tid}/turns",{
           **first,"branch_id":bid,"parent_message_id":head,"expected_version":0,
           "idempotency_key":"fork-A","content":"alternate direction"})
        self.assertEqual(status,200,new)
        status,other=self.call("POST",f"/v1/threads/{tid}/turns",{
           **first,"parent_message_id":head,"expected_version":1,
           "idempotency_key":"main-B","content":"original direction"})
        self.assertEqual(status,200,other)
        status,original=self.call("GET",f"/v1/threads/{tid}")
        self.assertEqual(original["head_message_id"],other["message_id"])
        status,fork_view=self.call("GET",f"/v1/threads/{tid}/branches/{bid}")
        self.assertEqual(fork_view["head_message_id"],new["message_id"])
        status,history=self.call("GET",f"/v1/threads/{tid}/branches/{bid}/messages")
        self.assertEqual([x["content"] for x in history["data"] if x["role"]=="user"],
                         ["root","alternate direction"])
        status,main_history=self.call("GET",f"/v1/threads/{tid}/messages")
        self.assertEqual([x["content"] for x in main_history["data"] if x["role"]=="user"],
                         ["root","original direction"])
        status,_=self.call("POST",f"/v1/threads/{tid}/branches",
                           {"from_message_id":"00000000-0000-0000-0000-000000000000"})
        self.assertEqual(status,404)

    def test_recovery_retry_preserves_single_logical_exchange(self):
        code,t=self.call("POST","/v1/threads",{"model":"fake-model"})
        self.assertEqual(code,201,t)
        tid=t["id"]; branch=t["branch_id"]
        msg={"branch_id":branch,"expected_version":0,"parent_message_id":None,
             "idempotency_key":"recover-1","content":"fail once","max_tokens":32}
        status,failed=self.call("POST",f"/v1/threads/{tid}/turns",msg)
        self.assertEqual(status,502,failed)
        status,empty=self.call("GET",f"/v1/threads/{tid}/messages")
        self.assertEqual((status,len(empty["data"])),(200,0))
        status,good=self.call("POST",f"/v1/threads/{tid}/turns",msg)
        self.assertEqual(status,200,good)
        self.assertEqual(good["status"],"COMMITTED")
        status,replay=self.call("POST",f"/v1/threads/{tid}/turns",msg)
        self.assertEqual(status,200,replay)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["message_id"],good["message_id"])
        status,items=self.call("GET",f"/v1/threads/{tid}/messages")
        self.assertEqual(status,200,items)
        self.assertEqual(len(items["data"]),2)
        self.assertEqual([x["role"] for x in items["data"]],["user","assistant"])


if __name__=="__main__":
    unittest.main()
