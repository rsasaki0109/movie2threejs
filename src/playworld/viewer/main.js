// playworld viewer: walk around a captured gaussian-splat world and knock things over.
import * as THREE from "three";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { ConvexGeometry } from "three/addons/geometries/ConvexGeometry.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import RAPIER from "@dimforge/rapier3d-compat";
import { validateShot, cameraAt, orderedEvents } from "./record.js";

const statusEl = document.getElementById("status");
const startEl = document.getElementById("start");
const setStatus = (s) => { statusEl.textContent = s; };

const params = new URLSearchParams(location.search);
const base = (params.get("world") ?? ".").replace(/\/?$/, "/");
const recording = params.has("record");
// Colab's preview proxy can reject pointer lock even in a separate window.
const dragMode = params.get("controls") === "drag";

const PLAYER_RADIUS = 0.25;
const PLAYER_HALF = 0.5; // capsule half-height (without the caps)
const WALK_SPEED = 2.2;
const JUMP_SPEED = 4.0;
const GRAVITY = -9.81;
const STEP = 1 / 60;

async function main() {
  setStatus("loading world.json…");
  const response = await fetch(base + "world.json");
  if (!response.ok) throw new Error(`world HTTP ${response.status}`);
  const world = await response.json();
  let shot = null;
  if (recording) {
    const response = await fetch(params.get("record"));
    if (!response.ok) throw new Error(`shot HTTP ${response.status}`);
    shot = validateShot(await response.json());
    startEl.classList.add("hidden");
    document.getElementById("hud").classList.add("hidden");
    document.getElementById("cross").classList.add("hidden");
    // Static-hosting chrome must not appear in canvas screenshots.
    for (const el of document.querySelectorAll("header, footer, #actions, #touch")) {
      el.classList.add("hidden");
    }
  }
  await RAPIER.init();

  // --- rendering -----------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: recording });
  renderer.setPixelRatio(recording ? 1 : Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  document.body.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202024);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));
  const camera = new THREE.PerspectiveCamera(shot?.fov ?? world.player.fov ?? 70, innerWidth / innerHeight, 0.03, 500);
  const spark = new SparkRenderer({ renderer, autoUpdate: !recording, preUpdate: !recording, enableLod: !recording });
  if (recording) spark.minSortIntervalMs = 0;
  scene.add(spark);
  for (const patch of world.support_patches ?? []) {
    const shape = new THREE.Shape(patch.polygon_xz.map(([x, z]) => new THREE.Vector2(x, -z)));
    const geometry = new THREE.ShapeGeometry(shape);
    geometry.rotateX(-Math.PI / 2);
    const color = new THREE.Color().setRGB(...patch.color, THREE.SRGBColorSpace);
    const vertexColors = patch.vertex_colors?.length === patch.polygon_xz.length;
    if (vertexColors) {
      const positions = geometry.getAttribute('position');
      const colors = [];
      // ShapeGeometry may reverse contour order when triangulating the shape.
      // Match coordinates, so the observed colors stay on their actual corners.
      for (let i = 0; i < positions.count; i++) {
        const distances = patch.polygon_xz.map(([x, z]) =>
          (x - positions.getX(i)) ** 2 + (z - positions.getZ(i)) ** 2);
        const index = distances.indexOf(Math.min(...distances));
        colors.push(...new THREE.Color().setRGB(...patch.vertex_colors[index], THREE.SRGBColorSpace).toArray());
      }
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    }
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: vertexColors ? 0xffffff : color, vertexColors, side: THREE.DoubleSide,
    }));
    mesh.position.y = patch.y;
    scene.add(mesh);
  }
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  const align = new THREE.Matrix4().fromArray(world.align);
  const loading = [];
  // Optional reviewed photo planes; these are not reconstructed exterior geometry.
  for (const backdrop of world.photo_backdrops ?? []) {
    if (backdrop.vertices?.length !== 4 || backdrop.uv?.length !== 4 ||
        !backdrop.vertices.every(p => p.length === 3 && p.every(Number.isFinite)) ||
        !backdrop.uv.every(p => p.length === 2 && p.every(Number.isFinite))) {
      throw new Error('Photo backdrop requires four finite world vertices and UV pairs');
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(backdrop.vertices.flat(), 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(backdrop.uv.flat(), 2));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    const texture = await new THREE.TextureLoader().loadAsync(base + backdrop.texture);
    texture.colorSpace = THREE.SRGBColorSpace;
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      map: texture, side: THREE.DoubleSide, toneMapped: false,
    })));
  }
  const downloads = new Map();
  const splat = (url, matrix) => {
    const mesh = new SplatMesh({ url: base + url, onProgress: event => {
      downloads.set(url, event.loaded);
      const mb = [...downloads.values()].reduce((sum, bytes) => sum + bytes, 0) / 1e6;
      setStatus(`Loading room… ${mb.toFixed(1)} MB`);
    } });
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
    let interior = null;
    if (o.fill_color) {
      let geometry;
      if (o.visual_fill?.kind === 'bottom_cap') {
        const vertices = o.visual_fill.vertices;
        const shape = new THREE.Shape(vertices.map(([x, , z]) => new THREE.Vector2(x, -z)));
        geometry = new THREE.ShapeGeometry(shape);
        geometry.rotateX(-Math.PI / 2);
        geometry.translate(0, vertices[0][1], 0);
      } else {
        const vertices = o.visual_fill?.kind === 'convex_hull' ? o.visual_fill.vertices : o.hull;
        geometry = new ConvexGeometry(vertices.map(p => new THREE.Vector3(...p).multiplyScalar(.985)));
      }
      const color = new THREE.Color().setRGB(...o.fill_color, THREE.SRGBColorSpace);
      // Reviewed fills may cover a box bottom or only a chair's upper interior.
      // The inward-facing surface remains behind the photographed splats.
      interior = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color,
        side: o.visual_fill?.kind === 'bottom_cap' ? THREE.DoubleSide : THREE.BackSide }));
      scene.add(interior);
    }
    return { o, body, offset, interior, mesh: splat(o.splat, offset) };
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
  let touchActive = false;
  let dragActive = false;
  let dragging = false;
  let lastPointer = null;
  if (dragMode && !recording) {
    document.getElementById("hud").textContent = "Drag to look · WASD move · Space jump · F throw · E push · R reset · Esc pause";
    renderer.domElement.addEventListener("pointerdown", (event) => {
      if (!dragActive || event.button !== 0) return;
      dragging = true;
      lastPointer = [event.clientX, event.clientY];
      renderer.domElement.setPointerCapture(event.pointerId);
    });
    renderer.domElement.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const rotation = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
      rotation.y -= (event.clientX - lastPointer[0]) * .002;
      rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2,
        rotation.x - (event.clientY - lastPointer[1]) * .002));
      camera.quaternion.setFromEuler(rotation);
      lastPointer = [event.clientX, event.clientY];
    });
    for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
      renderer.domElement.addEventListener(event, () => { dragging = false; });
    }
  }
  const playButton = document.getElementById("play") ?? startEl;
  playButton.addEventListener("click", () => {
    if (!window.playworld?.ready) return;
    if (dragMode) {
      dragActive = true;
      startEl.classList.add("hidden");
    } else if (matchMedia("(pointer: coarse)").matches && document.getElementById("touch")) {
      touchActive = true;
      startEl.classList.add("hidden");
      document.getElementById("touch").classList.remove("hidden");
    } else controls.lock();
  });
  controls.addEventListener("lock", () => startEl.classList.add("hidden"));
  controls.addEventListener("unlock", () => startEl.classList.remove("hidden"));
  document.addEventListener("pointerlockerror", () => setStatus("Mouse capture failed. Open the demo directly in a new tab and try again."));
  const keys = new Set();
  controls.addEventListener("unlock", () => keys.clear());
  addEventListener("blur", () => { keys.clear(); dragging = false; });
  document.addEventListener("visibilitychange", () => { if (document.hidden) keys.clear(); });
  addEventListener("keydown", (e) => {
    if (recording) return;
    keys.add(e.code);
    if (e.code === "Escape") {
      controls.unlock();
      if (dragActive) { dragActive = false; dragging = false; keys.clear(); startEl.classList.remove("hidden"); }
    }
    if (e.code === "KeyF" && dragActive && !e.repeat) throwBall();
    if (e.code === "KeyC") toggleDebug();
    if (e.code === "KeyR") reset();
    if (e.code === "KeyE") push();
  });
  addEventListener("keyup", (e) => { if (!recording) keys.delete(e.code); });
  addEventListener("mousedown", () => { if (!recording && controls.isLocked) throwBall(); });

  const eye = () => camera.getWorldPosition(new THREE.Vector3());
  const look = () => camera.getWorldDirection(new THREE.Vector3());

  const balls = [];
  const ballGeo = new THREE.SphereGeometry(0.08, 16, 12);
  const ballMat = new THREE.MeshStandardMaterial({ color: 0xff5533, roughness: 0.4 });
  function throwBall(origin = eye(), direction = look()) {
    const p = origin.clone().addScaledVector(direction, 0.4);
    const v = direction.clone().multiplyScalar(12);
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

  function push(o = eye(), direction = look(), strength = 3) {
    const d = direction.clone();
    const hit = phys.castRay(new RAPIER.Ray(o, d), 4, true, undefined, undefined, playerCol);
    const body = hit?.collider.parent();
    if (!body || !body.isDynamic()) return null;
    const p = new RAPIER.Ray(o, d).pointAt(hit.timeOfImpact);
    const impulse = d.multiplyScalar(strength * body.mass());
    body.applyImpulseAtPoint(impulse, p, true);
    return objects.find(object => object.body === body)?.o.id ?? null;
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

  function syncMeshes() {
    const t = playerBody.translation();
    if (!window.playworld?.freeCamera) camera.position.set(t.x, t.y + eyeOffset, t.z); // free camera: scripted shots
    for (const { body, offset, mesh, interior } of objects) {
      const p = body.translation(), q = body.rotation();
      mesh.matrix.compose(tmpV.set(p.x, p.y, p.z), tmpQ.set(q.x, q.y, q.z, q.w), one).multiply(offset);
      mesh.matrixWorldNeedsUpdate = true;
      if (interior) { interior.position.copy(p); interior.quaternion.copy(q); }
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
  }
  camera.position.set(sx, standY + eyeOffset, sz);
  if (world.player.look_at) camera.lookAt(new THREE.Vector3().fromArray(world.player.look_at));

  setStatus(`loading ${1 + objects.length} splats…`);
  await Promise.all(loading);
  setStatus([`${world.objects.length} physical objects · ${world.colliders.length} static colliders`,
    ...(world.photo_backdrops ?? []).map(b => b.label ?? 'Source-photo plane')].join(' · '));
  window.playworld = { world, phys, objects, camera, controls, renderer, push, throwBall, reset, freeCamera: recording,
    setMoveKey(code, down) { if (down) keys.add(code); else keys.delete(code); },
    lookBy(dx, dy) {
      const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
      euler.y -= dx * .004;
      euler.x = Math.max(-Math.PI / 2 + .05, Math.min(Math.PI / 2 - .05, euler.x - dy * .004));
      camera.quaternion.setFromEuler(euler);
    }
  };
  if (recording) {
    let tick = 0, frameIndex = 0, nextEvent = 0, busy = false;
    const events = orderedEvents(shot.events);
    const actions = [];
    function applyCamera(time) {
      const pose = cameraAt(shot.camera, time);
      camera.position.fromArray(pose.position);
      camera.lookAt(new THREE.Vector3().fromArray(pose.target));
    }
    function applyEvents() {
      while (nextEvent < events.length && events[nextEvent].time <= tick * STEP + 1e-9) {
        const event = events[nextEvent++];
        const origin = event.origin ? new THREE.Vector3().fromArray(event.origin) : eye();
        const direction = event.target ? new THREE.Vector3().fromArray(event.target).sub(origin).normalize() : look();
        if (event.type === "push") actions.push({ time: tick * STEP, type: "push", object: push(origin, direction, event.strength ?? 3) });
        if (event.type === "throw") { throwBall(origin, direction); actions.push({ time: tick * STEP, type: "throw" }); }
        if (event.type === "walk") { keys.clear(); event.keys.forEach(k => keys.add(k)); }
      }
    }
    const snapshot = () => ({ tick, objects: objects.map(({ o, body }) => ({ id: o.id, position: body.translation(), rotation: body.rotation() })), player: playerBody.translation(), balls: balls.map(({body}) => body.translation()), actions: [...actions] });
    async function renderRecorded() {
      syncMeshes();
      scene.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      await spark.update({ scene, camera }); // includes asynchronous depth sorting
      renderer.render(scene, camera);
      renderer.getContext().finish();
    }
    applyCamera(0);
    await renderRecorded();
    window.playworld.record = {
      fps: shot.fps, frames: Math.round(shot.duration * shot.fps), snapshot,
      async nextFrame(requestedFrame = frameIndex) {
        if (busy) throw new Error("record frames must be requested sequentially");
        if (!Number.isInteger(requestedFrame) || requestedFrame < frameIndex || requestedFrame >= this.frames) throw new Error("frame must advance within the shot");
        busy = true;
        try {
          const targetTick = requestedFrame * (60 / shot.fps);
          while (tick < targetTick) {
            applyCamera(tick * STEP);
            applyEvents();
            if (keys.size) stepPlayer(STEP);
            phys.step();
            tick++;
          }
          applyCamera(tick * STEP);
          applyEvents();
          await renderRecorded();
          frameIndex = requestedFrame + 1;
          return { frame: requestedFrame, time: tick * STEP, ...snapshot() };
        } finally { busy = false; }
      }
    };
  } else {
    // Show the fully sorted room before enabling entry, including slow GPUs.
    syncMeshes();
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    await spark.update({ scene, camera });
    renderer.render(scene, camera);
    last = performance.now();
    function frame(now) {
      acc += Math.min((now - last) / 1000, 0.1);
      last = now;
      while (acc >= STEP) {
        if (controls.isLocked || touchActive || dragActive) stepPlayer(STEP);
        phys.step();
        acc -= STEP;
      }
      syncMeshes();
      renderer.render(scene, camera);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  window.playworld.ready = true;
  if (playButton instanceof HTMLButtonElement) playButton.disabled = false;
  dispatchEvent(new Event("playworldready"));
}

main().catch((e) => { console.error(e); window.playworldError = e.stack ?? e.message; setStatus("error: " + e.message); });
