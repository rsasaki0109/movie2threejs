// Timeline utilities are independent of WebGL, so invalid shots fail before capture.
export function validateShot(shot) {
  if (shot.version !== 1) throw new Error("shot version must be 1");
  if (![15, 20, 30, 60].includes(shot.fps)) throw new Error("fps must divide 60 (15, 20, 30 or 60)");
  if (!Number.isFinite(shot.duration) || shot.duration <= 0) throw new Error("duration must be positive");
  if (!Array.isArray(shot.camera) || shot.camera.length < 2) throw new Error("at least two camera keyframes are required");
  let previous = -1;
  for (const key of shot.camera) {
    if (!Number.isFinite(key.time) || key.time <= previous || key.time > shot.duration) throw new Error("camera times must increase within duration");
    for (const name of ["position", "target"]) {
      if (!Array.isArray(key[name]) || key[name].length !== 3 || !key[name].every(Number.isFinite)) throw new Error(`${name} must be a finite xyz vector`);
    }
    if (key.position.every((v, i) => v === key.target[i])) throw new Error("camera target must differ from position");
    previous = key.time;
  }
  if (shot.camera[0].time !== 0 || previous !== shot.duration) throw new Error("camera must cover time 0 through duration");
  for (const event of shot.events ?? []) {
    if (!Number.isFinite(event.time) || event.time < 0 || event.time >= shot.duration) throw new Error("event time outside shot");
    if (!["push", "throw", "walk"].includes(event.type)) throw new Error(`unknown event ${event.type}`);
    if (event.type === "walk" && (!Array.isArray(event.keys) || !event.keys.every(k => ["KeyW", "KeyA", "KeyS", "KeyD", "Space", "ShiftLeft"].includes(k)))) throw new Error("invalid walk keys");
  }
  return shot;
}

export function cameraAt(keys, time) {
  let index = 0;
  while (index < keys.length - 2 && keys[index + 1].time < time) index++;
  const a = keys[index], b = keys[index + 1];
  let u = Math.max(0, Math.min(1, (time - a.time) / (b.time - a.time)));
  if (a.ease !== "linear") u = u * u * (3 - 2 * u);
  const mix = name => a[name].map((v, i) => v + (b[name][i] - v) * u);
  return { position: mix("position"), target: mix("target") };
}

export function orderedEvents(events) {
  return [...(events ?? [])].sort((a, b) => a.time - b.time);
}
