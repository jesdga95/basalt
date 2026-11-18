"""Responses release inference on interrupted exits, including a counted prompt before SSE headers."""
import http.client
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serve.api.responses import Assembler
from serve.engine.errors import ClientGone, EngineDied, EngineError
from serve.frontend import Event
from serve.response_generation import generate
from serve.tests.test_responses import Server
from serve.tests.test_text_service import TextEngine


class Primary:
    """A retained iterator: dropping the consumer cannot implicitly close its resources."""
    def __init__(self, values=(), *, error=None, close_error=None):
        self.values = iter(values)
        self.error, self.close_error = error, close_error
        self.closed = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.values)
        except StopIteration:
            if self.error:
                raise self.error
            raise

    def close(self):
        try:
            if hasattr(self.values, "close"):
                self.values.close()
            if self.close_error:
                raise self.close_error
        finally:
            self.closed.set()


class Generation(unittest.TestCase):
    def start(self, primary, *, cancel=None, session=None, live=None):
        self.cancel = cancel if cancel is not None else threading.Event()
        self.svc = SimpleNamespace(run=Mock(return_value=primary))
        self.asm = Assembler({"input": "hi"}, "test", 2, {}, False)
        return generate(self.svc, [1, 2], False, None, 16, {}, self.cancel, None, self.asm,
                        session=session, live=live)

    def test_close_during_keepalive_cancels_and_releases_retained_iterator(self):
        cancel = threading.Event()

        def values():
            try:
                yield "ping", None
            finally:
                self.assertTrue(cancel.is_set())

        primary = Primary(values())
        stream = self.start(primary, cancel=cancel)
        self.addCleanup(stream.close)
        for event in stream:
            if event is None:
                break
        stream.close()
        self.assertTrue(self.cancel.is_set())
        self.assertTrue(primary.closed.is_set())

    def test_primary_is_released_before_success_is_announced(self):
        primary = Primary([("event", Event("content", "ok")),
                           ("done", {"finish": "stop", "completion_tokens": 1, "prompt_tokens": 7})])
        live = object()
        events = []
        for event in self.start(primary, session="session", live=live):
            if event["type"] == "response.completed":
                self.assertTrue(primary.closed.is_set())
            events.append(event)
        self.assertFalse(self.cancel.is_set())
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["usage"]["input_tokens"], 7)
        self.assertEqual(self.svc.run.call_args.kwargs, {"force": None, "session": "session", "live": live})
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))

    def test_failure_and_close_failure_never_announce_success(self):
        for error, close_error in ((EngineDied("engine died"), None),
                                   (ValueError("generation failed"), None),
                                   (ValueError("generation failed"), EngineError("close failed")),
                                   (None, EngineError("close failed"))):
            with self.subTest(error=error, close_error=close_error):
                primary = Primary([("event", Event("content", "partial"))],
                                  error=error, close_error=close_error)
                emitted = []
                with self.assertRaisesRegex(type(close_error or error), str(close_error or error)):
                    for event in self.start(primary):
                        emitted.append(event)
                self.assertTrue(primary.closed.is_set())
                self.assertTrue(self.cancel.is_set())
                self.assertNotIn("response.completed", [e["type"] for e in emitted])

    def test_missing_outcome_is_an_engine_error(self):
        primary = Primary([("event", Event("content", "partial"))])
        with self.assertRaisesRegex(EngineError, "without an outcome"):
            list(self.start(primary))
        self.assertTrue(primary.closed.is_set())

    def test_engine_cancellation_without_disconnect_is_an_error(self):
        primary = Primary([("done", {"finish": "cancel", "completion_tokens": 0})])
        events = []
        with self.assertRaisesRegex(EngineError, "cancelled without a client disconnect"):
            for event in self.start(primary):
                events.append(event)
        self.assertTrue(primary.closed.is_set())
        self.assertTrue(self.cancel.is_set())
        self.assertNotIn("response.completed", [e["type"] for e in events])

    def test_cancelled_request_after_inference_exits_quietly(self):
        for finish in (None, "cancel", "stop"):
            with self.subTest(finish=finish):
                cancel = threading.Event()

                def values():
                    yield "event", Event("content", "partial")
                    cancel.set()
                    if finish is not None:
                        yield "done", {"finish": finish, "completion_tokens": 1}

                primary = Primary(values())
                events = []
                with self.assertRaises(ClientGone):
                    for event in self.start(primary, cancel=cancel):
                        events.append(event)
                self.assertTrue(primary.closed.is_set())
                self.assertNotIn("response.completed", [e["type"] for e in events])
                self.assertNotIn("response.incomplete", [e["type"] for e in events])

    def test_cancelled_request_does_not_start_inference(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(ClientGone):
            list(self.start(Primary(), cancel=cancel))
        self.svc.run.assert_not_called()

    def test_cancel_after_start_events_does_not_start_inference(self):
        stream = self.start(Primary())
        self.addCleanup(stream.close)
        self.assertEqual(next(stream)["type"], "response.created")
        self.cancel.set()
        events = []
        with self.assertRaises(ClientGone):
            for event in stream:
                events.append(event)
        self.assertNotIn("response.completed", [e["type"] for e in events])
        self.svc.run.assert_not_called()


class RecoveryHttp(Server):
    def test_cancel_after_start_or_inference_does_not_return_an_unfinished_response(self):
        start = Assembler.start
        for during_inference in (False, True):
            for stream in (False, True):
                with self.subTest(during_inference=during_inference, stream=stream):
                    cancel = threading.Event()

                    def events(asm):
                        yield from start(asm)
                        if not during_inference:
                            cancel.set()

                    def values():
                        yield "event", Event("content", "partial")
                        cancel.set()
                        yield "done", {"finish": "cancel", "completion_tokens": 1}

                    primary = Primary(values())
                    with patch.object(Assembler, "start", events), \
                         patch.object(self.httpd.RequestHandlerClass, "_client_cancel", return_value=cancel), \
                         patch.object(self.httpd, "handle_error") as error, \
                         patch.object(self.svc, "run", return_value=primary) as run:
                        req = {"model": "basalt", "input": "hi", "stream": stream}
                        if stream:
                            status, response = self.post(req)
                            self.assertEqual(status, 200)
                            types = [e["type"] for e in response]
                            self.assertIn("response.created", types)
                            self.assertNotIn("response.completed", types)
                            self.assertNotIn("response.incomplete", types)
                            self.assertNotIn("response.failed", types)
                            if during_inference:
                                self.assertIn("response.output_text.delta", types)
                        else:
                            with self.assertRaises(http.client.RemoteDisconnected):
                                self.post(req)
                    error.assert_not_called()
                    if during_inference:
                        run.assert_called_once()
                        self.assertTrue(primary.closed.is_set())
                    else:
                        run.assert_not_called()
                    status, response = self.post({"model": "basalt", "input": "recover", "max_output_tokens": 8})
                    self.assertEqual(status, 200, response)

    def test_cancel_before_first_event_does_not_traceback_or_start_inference(self):
        for text_engine in (False, True):
            with self.subTest(text_engine=text_engine):
                if text_engine:
                    self.svc.engine = TextEngine(self.tok, "ok", max_context=4096)
                cancel = threading.Event()
                cancel.set()
                with patch.object(self.httpd.RequestHandlerClass, "_client_cancel", return_value=cancel), \
                     patch.object(self.httpd, "handle_error") as error, \
                     patch.object(self.svc, "run") as run:
                    with self.assertRaises(http.client.RemoteDisconnected):
                        self.post({"model": "basalt", "input": "hi", "max_output_tokens": 8})
                run.assert_not_called()
                error.assert_not_called()
                status, response = self.post({"model": "basalt", "input": "recover", "max_output_tokens": 8})
                self.assertEqual(status, 200, response)

    def test_header_disconnect_releases_primed_inference_and_next_request_runs(self):
        for text_engine in (False, True):
            with self.subTest(text_engine=text_engine):
                if text_engine:
                    self.svc.engine = TextEngine(self.tok, "ok", max_context=4096)
                run, retained = self.svc.run, []

                def tracked(*args, **kwargs):
                    primary = Primary(run(*args, **kwargs))
                    retained.append(primary)
                    return primary

                with patch.object(self.svc, "run", side_effect=tracked), \
                     patch.object(self.httpd.RequestHandlerClass, "_sse", side_effect=BrokenPipeError):
                    with self.assertRaises(http.client.RemoteDisconnected):
                        self.post({"model": "basalt", "input": "hi", "stream": True, "max_output_tokens": 8})
                if text_engine:
                    self.assertEqual(len(retained), 1)
                    self.assertTrue(retained[0].closed.wait(2))
                else:
                    self.assertEqual(retained, [])
                self.assertTrue(self.svc.fifo.acquire(blocking=False))
                self.svc.fifo.release()
                status, response = self.post({"model": "basalt", "input": "hi", "max_output_tokens": 8})
                self.assertEqual(status, 200, response)
                self.assertIn(response["status"], ("completed", "incomplete"))

    def test_missing_outcome_is_500_or_stream_failure_and_releases_iterator(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                primary = Primary([("event", Event("content", "partial"))])
                with patch.object(self.svc, "run", return_value=primary):
                    status, response = self.post({"model": "basalt", "input": "hi", "stream": stream})
                self.assertTrue(primary.closed.is_set())
                if stream:
                    self.assertEqual(status, 200)
                    self.assertEqual(response[-1]["type"], "response.failed")
                    self.assertEqual(response[-1]["response"]["output"][0]["status"], "incomplete")
                    self.assertEqual([e["sequence_number"] for e in response], list(range(len(response))))
                else:
                    self.assertEqual(status, 500)
                    self.assertEqual(response["error"]["code"], "server_error")
                status, response = self.post({"model": "basalt", "input": "recover", "max_output_tokens": 8})
                self.assertEqual(status, 200, response)

    def test_engine_cancellation_is_500_or_stream_failure_and_releases_iterator(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                primary = Primary([("event", Event("content", "partial")),
                                   ("done", {"finish": "cancel", "completion_tokens": 1})])
                cancel = threading.Event()
                with patch.object(self.httpd.RequestHandlerClass, "_client_cancel", return_value=cancel), \
                     patch.object(self.svc, "run", return_value=primary):
                    status, response = self.post({"model": "basalt", "input": "hi", "stream": stream})
                self.assertTrue(primary.closed.is_set())
                self.assertTrue(cancel.is_set())
                if stream:
                    self.assertEqual(status, 200)
                    self.assertEqual(response[-1]["type"], "response.failed")
                    self.assertEqual(response[-1]["response"]["output"][0]["status"], "incomplete")
                    error = response[-1]["response"]["error"]
                    self.assertNotIn("response.completed", [e["type"] for e in response])
                    self.assertEqual([e["sequence_number"] for e in response], list(range(len(response))))
                else:
                    self.assertEqual(status, 500)
                    error = response["error"]
                self.assertEqual(error["code"], "server_error")
                self.assertIn("cancelled without a client disconnect", error["message"])
                status, response = self.post({"model": "basalt", "input": "recover", "max_output_tokens": 8})
                self.assertEqual(status, 200, response)


if __name__ == "__main__":
    unittest.main()
