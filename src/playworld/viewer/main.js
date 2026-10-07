// playworld viewer: walk around a captured gaussian-splat world and knock things over.
import * as THREE from "three";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import RAPIER from "@dimforge/rapier3d-compat";

const statusEl = document.getElementById("status");
const startEl = document.getElementById("start");
const setStatus = (s) => { statusEl.textContent = s; };

const params = new URLSearchParams(location.search);
const base = (params.get("world") ?? ".").replace(/\/?$/, "/");

const PLAYER_RADIUS = 0.25;
const PLAYER_HALF = 0.5; // capsule half-height (without the caps)
const WALK_SPEED = 2.2;
const JUMP_SPEED = 4.0;
const GRAVITY = -9.81;
const STEP = 1 / 60;

async function main() {
  setStatus("loading world.json…");
  const world = await (await fetch(base + "world.json")).json();
  await RAPIER.init();

  // --- rendering -----------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  document.body.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202024);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));
  const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.03, 500);
  scene.add(new SparkRenderer({ renderer }));
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  const align = new THREE.Matrix4().fromArray(world.align);
  const loading = [];
  const splat = (url, matrix) => {
    const mesh = new SplatMesh({ url: base + url });
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(matrix);
    mesh.matrixWorldNeedsUpdate = true;
    scene.add(mesh);
    if (mesh.initialized) loading.push(mesh.initialized);
    return mesh;
  };
  splat(world.background, align);

  // --- physics -------------------------------------------------------------
  const phys = new RAPIER.World({ x: 0, y: GRAVITY, z: 0 });
  phys.timestep = STEP;
  for (const c of world.colliders) {
    const [hx, hy, hz] = c.half_extents;
    const [x, y, z] = c.center;
    phys.createCollider(RAPIER.ColliderDesc.cuboid(hx, hy, hz).setTranslation(x, y, z));
  }

  // Each object: a dynamic convex hull whose pose drives its splat.
  // splat matrix = bodyPose * translate(-centroid) * align (splats stay in capture coordinates).
  const objects = world.objects.map((o) => {
    const [cx, cy, cz] = o.centroid;
    const body = phys.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(cx, cy, cz).setCanSleep(true));
    const desc = RAPIER.ColliderDesc.convexHull(new Float32Array(o.hull.flat()))
      ?? RAPIER.ColliderDesc.ball(0.1);
    phys.createCollider(desc.setMass(o.mass).setFriction(0.8), body);
    const offset = new THREE.Matrix4().makeTranslation(-cx, -cy, -cz).multiply(align);
    return { o, body, offset, mesh: splat(o.splat, offset) };
  });

  // Player: kinematic capsule moved by Rapier's character controller (pushes dynamic bodies too).
  const [sx, , sz] = world.player.spawn;
  const standY = PLAYER_HALF + PLAYER_RADIUS + 0.05;
  const playerBody = phys.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(sx, standY, sz));
  const playerCol = phys.createCollider(RAPIER.ColliderDesc.capsule(PLAYER_HALF, PLAYER_RADIUS), playerBody);
  const controller = phys.createCharacterController(0.02);
  controller.enableAutostep(0.25, 0.15, false);
  controller.enableSnapToGround(0.25);
  controller.setApplyImpulsesToDynamicBodies(true);
  const eyeOffset = world.player.eye_height - (PLAYER_HALF + PLAYER_RADIUS);
  let vy = 0;

  // --- input ---------------------------------------------------------------
  const controls = new PointerLockControls(camera, document.body);
  startEl.addEventListener("click", () => controls.lock());
  controls.addEventListener("lock", () => startEl.classList.add("hidden"));
  controls.addEventListener("unlock", () => startEl.classList.remove("hidden"));
  const keys = new Set();
  addEventListener("keydown", (e) => {
    keys.add(e.code);
    if (e.code === "KeyC") toggleDebug();
    if (e.code === "KeyR") reset();
    if (e.code === "KeyE") push();
  });
  addEventListener("keyup", (e) => keys.delete(e.code));
  addEventListener("mousedown", () => { if (controls.isLocked) throwBall(); });

  const eye = () => camera.getWorldPosition(new THREE.Vector3());
  const look = () => camera.getWorldDirection(new THREE.Vector3());

  const balls = [];
  const ballGeo = new THREE.SphereGeometry(0.08, 16, 12);
  const ballMat = new THREE.MeshStandardMaterial({ color: 0xff5533, roughness: 0.4 });
  function throwBall() {
    const p = eye().addScaledVector(look(), 0.4);
    const v = look().multiplyScalar(12);
    const body = phys.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(p.x, p.y, p.z).setLinvel(v.x, v.y, v.z).setCcdEnabled(true));
    phys.createCollider(RAPIER.ColliderDesc.ball(0.08).setMass(0.4).setRestitution(0.4), body);
    const mesh = new THREE.Mesh(ballGeo, ballMat);
    scene.add(mesh);
    balls.push({ body, mesh });
    if (balls.length > 40) {
      const b = balls.shift();
      phys.removeRigidBody(b.body);
      scene.remove(b.mesh);
    }
  }

  function push() {
    const o = eye(), d = look();
    const hit = phys.castRay(new RAPIER.Ray(o, d), 4, true, undefined, undefined, playerCol);
    const body = hit?.collider.parent();
    if (!body || !body.isDynamic()) return;
    const p = new RAPIER.Ray(o, d).pointAt(hit.timeOfImpact);
    const impulse = d.multiplyScalar(3 * body.mass());
    body.applyImpulseAtPoint(impulse, p, true);
  }

  function reset() {
    for (const { o, body } of objects) {
      body.setTranslation({ x: o.centroid[0], y: o.centroid[1], z: o.centroid[2] }, true);
      body.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    }
    for (const b of balls.splice(0)) { phys.removeRigidBody(b.body); scene.remove(b.mesh); }
    playerBody.setNextKinematicTranslation({ x: sx, y: standY, z: sz });
    playerBody.setTranslation({ x: sx, y: standY, z: sz }, true);
    vy = 0;
  }

  let debugLines = null;
  function toggleDebug() {
    if (debugLines) { scene.remove(debugLines); debugLines = null; return; }
    debugLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }));
    scene.add(debugLines);
  }
  if (params.has("debug")) toggleDebug();

  // --- loop ----------------------------------------------------------------
  const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpV = new THREE.Vector3(), one = new THREE.Vector3(1, 1, 1);
  let acc = 0, last = performance.now();
  function stepPlayer(dt) {
    const fwd = look(); fwd.y = 0; fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
    const move = new THREE.Vector3();
    if (keys.has("KeyW")) move.add(fwd);
    if (keys.has("KeyS")) move.sub(fwd);
    if (keys.has("KeyD")) move.add(right);
    if (keys.has("KeyA")) move.sub(right);
    if (move.lengthSq() > 0) move.normalize().multiplyScalar(WALK_SPEED * (keys.has("ShiftLeft") ? 2 : 1) * dt);
    vy += GRAVITY * dt;
    move.y = vy * dt;
    controller.computeColliderMovement(playerCol, move);
    const m = controller.computedMovement();
    if (controller.computedGrounded()) {
      vy = keys.has("Space") ? JUMP_SPEED : 0;
    }
    const t = playerBody.translation();
    playerBody.setNextKinematicTranslation({ x: t.x + m.x, y: t.y + m.y, z: t.z + m.z });
    if (t.y < -20) reset();
  }

  function frame(now) {
    acc += Math.min((now - last) / 1000, 0.1);
    last = now;
    while (acc >= STEP) {
      if (controls.isLocked) stepPlayer(STEP);
      phys.step();
      acc -= STEP;
    }
    const t = playerBody.translation();
    if (!window.playworld?.freeCamera) camera.position.set(t.x, t.y + eyeOffset, t.z); // free camera: scripted shots
    for (const { body, offset, mesh } of objects) {
      const p = body.translation(), q = body.rotation();
      mesh.matrix.compose(tmpV.set(p.x, p.y, p.z), tmpQ.set(q.x, q.y, q.z, q.w), one).multiply(offset);
      mesh.matrixWorldNeedsUpdate = true;
    }
    for (const { body, mesh } of balls) {
      mesh.position.copy(body.translation());
      mesh.quaternion.copy(body.rotation());
    }
    if (debugLines) {
      const { vertices, colors } = phys.debugRender();
      debugLines.geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
      debugLines.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 4));
    }
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  camera.position.set(sx, standY + eyeOffset, sz);
  requestAnimationFrame(frame);

  setStatus(`loading ${1 + objects.length} splats…`);
  await Promise.all(loading);
  setStatus(`${world.objects.length} physical objects · ${world.colliders.length} static colliders`);
  window.playworld = { world, phys, objects, camera, controls }; // handy for debugging and tests
}

main().catch((e) => { console.error(e); setStatus("error: " + e.message); });
