import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/job-notifications.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const { JobFailureNotifications } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

function job(id, status, revision = 1) {
  return { id, status, revision, kind: "gallery", payload: {}, message: status,
    error: status === "failed" ? "requires bitsandbytes" : undefined };
}

test("opening the app does not present saved failures as new runtime errors", () => {
  const tracker = new JobFailureNotifications();
  const history = [job("old", "failed", 5)];
  assert.deepEqual(tracker.consume(history), []);
  assert.deepEqual(tracker.consume(history), []);
  assert.equal(history[0].error, "requires bitsandbytes");
});

test("a newly failed running job alerts even when it is not the newest job", () => {
  const tracker = new JobFailureNotifications();
  tracker.consume([job("newer", "completed"), job("running", "running")]);
  assert.deepEqual(tracker.consume([job("newer", "completed"), job("running", "failed", 2)]).map(j => j.id), ["running"]);
});

test("unrelated job changes do not resurrect an old dismissed error", () => {
  const tracker = new JobFailureNotifications();
  tracker.consume([job("old", "failed", 5)]);
  assert.deepEqual(tracker.consume([job("new", "running"), job("old", "failed", 5)]), []);
  assert.deepEqual(tracker.consume([job("new", "completed", 2), job("old", "failed", 6)]), []);
});

test("newly submitted jobs failing between snapshots are still reported", () => {
  const tracker = new JobFailureNotifications();
  tracker.consume([]);
  assert.equal(tracker.consume([job("new", "failed", 3)]).length, 1);
});

test("repeated stream snapshots and reconnects do not repeat failure alerts", () => {
  const tracker = new JobFailureNotifications();
  tracker.consume([job("new", "running")]);
  assert.equal(tracker.consume([job("new", "failed", 2)]).length, 1);
  assert.deepEqual(tracker.consume([job("new", "failed", 2)]), []);
  assert.deepEqual(tracker.consume([job("new", "failed", 3)]), []);
});

test("resuming a saved failure can produce a fresh failure notification", () => {
  const tracker = new JobFailureNotifications();
  tracker.consume([job("saved", "failed", 5)]);
  assert.deepEqual(tracker.consume([job("saved", "queued", 6)]), []);
  assert.equal(tracker.consume([job("saved", "failed", 8)]).length, 1);
});

test("out-of-order polling cannot reset a failure and make it alert twice", () => {
  const tracker = new JobFailureNotifications();
  tracker.consume([job("new", "running", 1)]);
  assert.equal(tracker.consume([job("new", "failed", 3)]).length, 1);
  assert.deepEqual(tracker.consume([job("new", "running", 2)]), []);
  assert.deepEqual(tracker.consume([job("new", "failed", 3)]), []);
});

test("completed, cancelled and interrupted jobs are not new failure alerts", () => {
  const tracker = new JobFailureNotifications();
  tracker.consume([]);
  assert.deepEqual(tracker.consume([job("a", "completed"), job("b", "cancelled"), job("c", "interrupted")]), []);
});
