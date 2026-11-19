"""Responses release inference on interrupted exits, including a counted prompt before SSE headers."""
import http.client
import json
import socket
import unittest
from unittest.mock import patch

from serve.prompt import primed
from serve.tests.test_responses import Server
from serve.tests.test_text_service import TextEngine


class RecoveryHttp(Server):
    def test_collected_response_remains_readable_after_client_half_closes(self):
        generate = self.engine.generate

        def wait_for_cancel(ids, max_new, sampling, cancel, embeddings=None, session=None):
            if not cancel.wait(5):
                yield from generate(ids, max_new, sampling, cancel, embeddings, session)

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        body = {"model": "basalt", "input": "hi", "max_output_tokens": 8, "reasoning": {"effort": "minimal"}}
        with patch.object(self.engine, "generate", side_effect=wait_for_cancel):
            conn.request("POST", "/v1/responses", json.dumps(body), {"Content-Type": "application/json"})
            conn.sock.shutdown(socket.SHUT_WR)
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)["status"], "in_progress")

    def test_header_disconnect_releases_primed_inference_and_next_request_runs(self):
        self.svc.engine = TextEngine(self.tok, "ok", max_context=4096)
        busy_at_headers = []
        run = self.svc.run
        retained = []

        def retain_inference(*args, **kwargs):
            inference = run(*args, **kwargs)
            retained.append(inference)
            return inference

        def fail_headers():
            with self.svc.status_lock:
                busy_at_headers.append(self.svc.status["busy"])
            raise BrokenPipeError

        # Retain the primed source and the service iterator so both must be closed explicitly.
        with patch("serve.handler.primed", wraps=primed) as prime, \
             patch.object(self.svc, "run", side_effect=retain_inference), \
             patch.object(self.httpd.RequestHandlerClass, "end_headers", side_effect=fail_headers), \
             patch.object(self.httpd, "handle_error") as error:
            with self.assertRaises(http.client.RemoteDisconnected):
                self.post({"model": "basalt", "input": "hi", "stream": True, "max_output_tokens": 8})
        self.assertEqual(len(retained), 1)
        self.addCleanup(retained[0].close)
        self.addCleanup(prime.call_args.args[0].close)
        self.assertEqual(busy_at_headers, [True])
        error.assert_not_called()
        self.assertTrue(self.svc.fifo.acquire(blocking=False))
        self.svc.fifo.release()
        status, response = self.post({"model": "basalt", "input": "hi", "max_output_tokens": 8})
        self.assertEqual(status, 200, response)
        self.assertIn(response["status"], ("completed", "incomplete"))


if __name__ == "__main__":
    unittest.main()
