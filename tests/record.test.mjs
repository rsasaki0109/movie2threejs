import test from "node:test";
import assert from "node:assert/strict";
import { validateShot, cameraAt, orderedEvents } from "../src/playworld/viewer/record.js";

const shot = () => ({version: 1, fps: 30, duration: 2, camera: [
  {time: 0, position: [0,1,2], target: [0,0,0]},
  {time: 2, position: [2,1,2], target: [0,0,0]}
], events: [{time: 1, type: "push"}]});

test("interpolation clamps to endpoints and passes through midpoint", () => {
  const keys = shot().camera;
  assert.deepEqual(cameraAt(keys, -1).position, [0,1,2]);
  assert.deepEqual(cameraAt(keys, 1).position, [1,1,2]);
  assert.deepEqual(cameraAt(keys, 3).position, [2,1,2]);
});

test("timeline rejects unsupported fps, unordered cameras and invalid events", () => {
  assert.equal(validateShot(shot()).fps, 30);
  const bad = shot(); bad.fps = 24;
  assert.throws(() => validateShot(bad), /fps/);
  bad.fps = 30; bad.camera[1].time = 0;
  assert.throws(() => validateShot(bad), /increase/);
  const outside = shot(); outside.events[0].time = 2;
  assert.throws(() => validateShot(outside), /outside/);
});

test("same-time events preserve JSON order", () => {
  const events = [{time: 1, type: "throw"}, {time: 0, type: "walk", keys: ["KeyW"]}, {time: 1, type: "push"}];
  assert.deepEqual(orderedEvents(events).map(e => e.type), ["walk", "throw", "push"]);
});
