import {
  DrawingUtils,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import {
  createFaceLandmarker,
  drawFaceMesh,
  getFaceBounds,
} from "./face-tracking.js";
import {
  classifyGesture,
  createHandLandmarker,
  drawHands,
} from "./hand-tracking.js";
import {
  classifyPose,
  createPoseLandmarker,
  drawPose,
} from "./pose-tracking.js";

// Configure model limits and the tracking update rate.
const MAX_FACES = 5;
const MAX_HANDS = 8;
const INFERENCE_INTERVAL_MS = 1000 / 15;
const FACE_TRACK_GRACE_MS = 750;
const FACE_MATCH_DISTANCE = 0.18;
const HAND_WRIST_MATCH_DISTANCE = 0.2;
const POINT_COOLDOWN_MS = 1200;
const FACE_NUMBER_STORAGE_KEY = "face-game-next-number";
const POINT_SETTINGS_STORAGE_KEY = "face-game-point-settings";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Cache the page controls and display elements used by the tracking workflow.
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const statusElement = document.querySelector("#status");
const faceCountElement = document.querySelector("#face-count");
const handCountElement = document.querySelector("#hand-count");
const poseCountElement = document.querySelector("#pose-count");
const video = document.querySelector("#camera");
const meshCanvas = document.querySelector("#tracking-overlay");
const meshContext = meshCanvas.getContext("2d");
const faceLabels = document.querySelector("#face-labels");
const trackedFaceList = document.querySelector("#tracked-face-list");
const pointControls = [
  { key: "fist", input: document.querySelector("#points-fist"), output: document.querySelector("#points-fist-value") },
  { key: "raised-fist", input: document.querySelector("#points-raised-fist"), output: document.querySelector("#points-raised-fist-value") },
  { key: "middle-finger", input: document.querySelector("#points-middle-finger"), output: document.querySelector("#points-middle-finger-value") },
  { key: "both-arms-raised", input: document.querySelector("#points-both-arms-raised"), output: document.querySelector("#points-both-arms-raised-value") },
];

// Keep detections and points in memory; storage contains only a number counter and settings.
let faceLandmarker;
let handLandmarker;
let poseLandmarker;
const drawingUtils = new DrawingUtils(meshContext);
let cameraStream;
let animationFrameId;
let lastInferenceAt = 0;
let faceTracks = [];
let nextFaceNumber = readNextFaceNumber();
const sessionScores = new Map();

function readNextFaceNumber() {
  try {
    const storedNumber = Number(localStorage.getItem(FACE_NUMBER_STORAGE_KEY));
    return Number.isSafeInteger(storedNumber) && storedNumber > 0 ? storedNumber : 1;
  } catch {
    return 1;
  }
}

function saveNextFaceNumber() {
  try {
    localStorage.setItem(FACE_NUMBER_STORAGE_KEY, String(nextFaceNumber));
  } catch {
    // Numbering still works for the current page if storage is unavailable.
  }
}

function loadPointSettings() {
  let savedSettings = {};
  try {
    const parsedSettings = JSON.parse(localStorage.getItem(POINT_SETTINGS_STORAGE_KEY) || "{}");
    if (parsedSettings && typeof parsedSettings === "object" && !Array.isArray(parsedSettings)) {
      savedSettings = parsedSettings;
    }
  } catch {
    savedSettings = {};
  }

  for (const control of pointControls) {
    const savedValue = Number(savedSettings[control.key]);
    if (Number.isFinite(savedValue)) {
      control.input.value = String(Math.max(0, Math.min(10, Math.round(savedValue))));
    }
    control.output.value = `${control.input.value} pts`;
  }
}

function savePointSettings() {
  try {
    const settings = Object.fromEntries(
      pointControls.map(({ key, input }) => [key, Number(input.value)]),
    );
    localStorage.setItem(POINT_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Controls remain usable for the current page if storage is unavailable.
  }
}

function updateFaceTracks(faceLandmarks, now) {
  const bounds = faceLandmarks.map(getFaceBounds);
  const availableTracks = faceTracks.filter(
    (track) => now - track.lastSeen <= FACE_TRACK_GRACE_MS,
  );
  const candidates = [];

  bounds.forEach((face, faceIndex) => {
    availableTracks.forEach((track) => {
      const distance = Math.hypot(face.centerX - track.centerX, face.centerY - track.centerY);
      if (distance <= FACE_MATCH_DISTANCE) candidates.push({ faceIndex, track, distance });
    });
  });
  candidates.sort((first, second) => first.distance - second.distance);

  const assigned = new Array(bounds.length);
  const assignedFaces = new Set();
  const assignedTracks = new Set();
  for (const candidate of candidates) {
    if (assignedFaces.has(candidate.faceIndex) || assignedTracks.has(candidate.track)) continue;
    assigned[candidate.faceIndex] = candidate.track;
    assignedFaces.add(candidate.faceIndex);
    assignedTracks.add(candidate.track);
  }

  bounds.forEach((face, faceIndex) => {
    let track = assigned[faceIndex];
    if (!track) {
      track = { id: nextFaceNumber++ };
      assigned[faceIndex] = track;
      availableTracks.push(track);
      saveNextFaceNumber();
    }
    track.centerX = face.centerX;
    track.centerY = face.centerY;
    track.lastSeen = now;
    track.bounds = face;
  });

  faceTracks = availableTracks;
  return assigned.map((track) => ({ id: track.id, bounds: track.bounds }));
}

function matchPosesToFaces(currentFaces, poses) {
  const candidates = [];
  currentFaces.forEach((face, faceIndex) => {
    poses.forEach((landmarks, poseIndex) => {
      const nose = landmarks[0];
      if (!nose || (nose.visibility ?? 1) < 0.25) return;
      const distance = Math.hypot(
        face.bounds.centerX - nose.x,
        face.bounds.centerY - nose.y,
      );
      if (distance <= 0.2) candidates.push({ faceIndex, poseIndex, distance });
    });
  }
  candidates.sort((first, second) => first.distance - second.distance);

  const poseByFace = new Map();
  const usedFaces = new Set();
  const usedPoses = new Set();
  for (const candidate of candidates) {
    if (usedFaces.has(candidate.faceIndex) || usedPoses.has(candidate.poseIndex)) continue;
    poseByFace.set(candidate.faceIndex, candidate.poseIndex);
    usedFaces.add(candidate.faceIndex);
    usedPoses.add(candidate.poseIndex);
  }
  return poseByFace;
}

function matchHandsToPoses(handLandmarks, poses) {
  const cuesByPose = poses.map(() => []);
  for (const hand of handLandmarks) {
    const cue = classifyGesture(hand);
    if (!cue) continue;

    const wrist = hand[0];
    let bestPoseIndex = -1;
    let bestDistance = HAND_WRIST_MATCH_DISTANCE;
    poses.forEach((pose, poseIndex) => {
      for (const wristIndex of [15, 16]) {
        const poseWrist = pose[wristIndex];
        if (!poseWrist || (poseWrist.visibility ?? 1) < 0.25) continue;
        const distance = Math.hypot(wrist.x - poseWrist.x, wrist.y - poseWrist.y);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestPoseIndex = poseIndex;
        }
      }
    });
    if (bestPoseIndex >= 0) cuesByPose[bestPoseIndex].push(cue);
  }
  return cuesByPose;
}

function getSessionScore(faceNumber) {
  if (!sessionScores.has(faceNumber)) {
    sessionScores.set(faceNumber, {
      points: 0,
      activeCueKeys: new Set(),
      lastAwardedAt: new Map(),
      cueLabels: [],
    });
  }
  return sessionScores.get(faceNumber);
}

function updateGameScores(currentFaces, poseResult, handResult, now) {
  const poses = poseResult.landmarks;
  const poseByFace = matchPosesToFaces(currentFaces, poses);
  const handCuesByPose = matchHandsToPoses(handResult.landmarks, poses);
  const visibleFaces = new Set();

  currentFaces.forEach((face, faceIndex) => {
    visibleFaces.add(face.id);
    const score = getSessionScore(face.id);
    const cues = new Map();
    const poseIndex = poseByFace.get(faceIndex);
    if (poseIndex !== undefined) {
      for (const cue of classifyPose(poses[poseIndex])) cues.set(cue.key, cue.label);
      for (const cue of handCuesByPose[poseIndex]) cues.set(cue.key, cue.label);
    }

    for (const key of cues.keys()) {
      if (score.activeCueKeys.has(key)) continue;
      const lastAwardedAt = score.lastAwardedAt.get(key) ?? -Infinity;
      if (now - lastAwardedAt < POINT_COOLDOWN_MS) continue;
      const control = pointControls.find((candidate) => candidate.key === key);
      if (!control) continue;
      score.points += Number(control.input.value);
      score.lastAwardedAt.set(key, now);
    }
    score.activeCueKeys = new Set(cues.keys());
    score.cueLabels = [...cues.values()];
  });

  for (const [faceNumber, score] of sessionScores) {
    if (visibleFaces.has(faceNumber)) continue;
    score.activeCueKeys.clear();
    score.cueLabels = [];
  }

  renderScoreboard(visibleFaces);
}

function renderScoreboard(visibleFaces) {
  trackedFaceList.replaceChildren();
  const sortedScores = [...sessionScores.entries()].sort(([first], [second]) => first - second);
  for (const [faceNumber, score] of sortedScores) {
    const item = document.createElement("li");
    const presence = visibleFaces.has(faceNumber) ? "" : " (out of frame)";
    const cues = score.cueLabels.length > 0 ? ` - ${score.cueLabels.join(", ")}` : "";
    item.textContent = `Face ${faceNumber}: ${score.points} points${presence}${cues}`;
    trackedFaceList.append(item);
  }
}

function renderFaceLabels(currentFaces) {
  const labels = currentFaces.map(({ id, bounds }) => {
    const label = document.createElement("span");
    label.className = "face-label";
    label.textContent = `Face ${id}`;
    label.style.left = `${Math.max(0, (1 - bounds.right) * 100)}%`;
    label.style.top = `${Math.max(0, bounds.top * 100)}%`;
    return label;
  });
  faceLabels.replaceChildren(...labels);
}

// Display a message in the page's accessible status area.
function setStatus(message) {
  statusElement.textContent = message;
}

// Run face, hand, and pose inference and update this session's game board.
function detectFrame() {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }

  const now = performance.now();
  if (now - lastInferenceAt < INFERENCE_INTERVAL_MS) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }
  lastInferenceAt = now;

  try {
    const faceResult = faceLandmarker.detectForVideo(video, now);
    const handResult = handLandmarker.detectForVideo(video, now);
    const poseResult = poseLandmarker.detectForVideo(video, now);
    faceCountElement.value = String(faceResult.faceLandmarks.length);
    handCountElement.value = String(handResult.landmarks.length);
    poseCountElement.value = String(poseResult.landmarks.length);
    const currentFaces = updateFaceTracks(faceResult.faceLandmarks, now);
    updateGameScores(currentFaces, poseResult, handResult, now);
    renderFaceLabels(currentFaces);
    drawFaceMesh(faceResult, video, meshCanvas, meshContext, drawingUtils);
    drawPose(poseResult, drawingUtils);
    drawHands(handResult, drawingUtils);

    if (!cameraStream) return;
  } catch (error) {
    setStatus(`Face, hand, or pose tracking failed: ${error.message}`);
    stopCamera();
    return;
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

// Request camera access, initialize any missing models, and start frame processing.
async function startCamera() {
  startButton.disabled = true;
  setStatus("Requesting camera access…");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access requires HTTPS (or localhost) and a supported browser. On iPhone, open this page in Safari and allow camera access.",
      );
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 24, max: 30 },
      },
    });
    video.srcObject = cameraStream;
    await video.play();

    if (!faceLandmarker || !handLandmarker || !poseLandmarker) {
      setStatus("Loading face, hand, and pose trackers…");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      if (!faceLandmarker) {
        faceLandmarker = await createFaceLandmarker(vision, MAX_FACES);
      }
      if (!handLandmarker) {
        handLandmarker = await createHandLandmarker(vision, MAX_HANDS);
      }
      if (!poseLandmarker) {
        poseLandmarker = await createPoseLandmarker(vision, MAX_FACES);
      }
    }
    stopButton.disabled = false;
    lastInferenceAt = 0;
    setStatus("Camera running. Face numbers and points last for this camera session.");
    animationFrameId = requestAnimationFrame(detectFrame);
  } catch (error) {
    stopCamera();
    setStatus(`Could not start camera or trackers: ${error.message}`);
  } finally {
    startButton.disabled = false;
  }
}

// Stop frame processing and camera tracks, then reset session IDs and points.
function stopCamera() {
  if (animationFrameId !== undefined) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = undefined;
  }
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
  meshContext.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  faceCountElement.value = "0";
  handCountElement.value = "0";
  poseCountElement.value = "0";
  faceTracks = [];
  sessionScores.clear();
  trackedFaceList.replaceChildren();
  faceLabels.replaceChildren();
  stopButton.disabled = true;
  startButton.disabled = false;
  if (statusElement.textContent.startsWith("Camera running.")) {
    setStatus("Camera stopped.");
  }
}

// Connect the page buttons and point sliders to their actions.
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

loadPointSettings();
for (const { input, output } of pointControls) {
  input.addEventListener("input", () => {
    output.value = `${input.value} pts`;
    savePointSettings();
  });
}
