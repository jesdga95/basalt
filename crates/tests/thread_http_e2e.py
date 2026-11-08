"""Integration contract against a real PostgreSQL and the Rust API with a fake Basalt engine.

Run after cargo build with DATABASE_URL pointing to a disposable test database.
"""
import json
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
    def log_message(self, *_args):
        pass
    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        self.seen.append(req)
        body = json.dumps({"choices":[{"message":{"role":"assistant","content":"reply-"+str(len(self.seen))}}]}).encode()
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
        self.assertEqual([x["content"] for x in history["data"]],
                         ["hello","reply-1","continue","reply-2"])
        self.assertEqual(len(FakeBasalt.seen),2)
        self.assertEqual(len(FakeBasalt.seen[-1]["messages"]),3)
        code,_=self.call("POST",f"/v1/threads/{tid}/turns",
                         {**second,"idempotency_key":"stale-key"})
        self.assertEqual(code,409)


if __name__=="__main__":
    unittest.main()
