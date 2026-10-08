import * as THREE from "three";

addEventListener("playworldready", () => {
  if (new URLSearchParams(location.search).has('record')) return;
  const world = window.playworld;
  const demo = world.world.demo ?? {};
  if (demo.title) document.querySelector('h1').textContent = demo.title;
  if (demo.description) document.querySelector('.lede').textContent = demo.description;
  if (demo.dataset) document.querySelector('footer a').textContent = `Eyeful Tower ${demo.dataset} · MIT data ↗`;
  if (demo.id) document.querySelector('.brand').href = '../../';
  const featured = document.getElementById("featured-push");
  const pushLabel = demo.push_label ?? 'Knock the box over';
  featured.textContent = pushLabel;
  featured.disabled = false;
  featured.hidden = !demo.push;
  document.getElementById("reset-world").disabled = false;
  featured.addEventListener("click", () => {
    const action = demo.push;
    const origin = new THREE.Vector3().fromArray(action.origin);
    const direction = new THREE.Vector3().fromArray(action.target).sub(origin).normalize();
    const id = world.push(origin, direction, action.strength);
    featured.textContent = id === null ? "Reset to try again" : "Pushed!";
  });
  document.getElementById("reset-world").addEventListener("click", () => {
    world.reset();
    world.camera.lookAt(new THREE.Vector3().fromArray(world.world.player.look_at));
    featured.textContent = pushLabel;
  });
  document.getElementById("touch-push").addEventListener("click", () => world.push());
  document.getElementById("touch-throw").addEventListener("click", () => world.throwBall());
  for (const button of document.querySelectorAll("[data-key]")) {
    button.addEventListener("pointerdown", e => {
      e.preventDefault(); button.setPointerCapture(e.pointerId);
      world.setMoveKey(button.dataset.key, true);
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      button.addEventListener(type, () => world.setMoveKey(button.dataset.key, false));
    }
  }
  let pointer = null;
  const canvas = world.renderer.domElement;
  canvas.addEventListener("pointerdown", e => {
    if (e.pointerType !== "touch") return;
    pointer = { id: e.pointerId, x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!pointer || pointer.id !== e.pointerId) return;
    world.lookBy(e.clientX - pointer.x, e.clientY - pointer.y);
    pointer.x = e.clientX; pointer.y = e.clientY;
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    canvas.addEventListener(type, () => { pointer = null; });
  }
});
