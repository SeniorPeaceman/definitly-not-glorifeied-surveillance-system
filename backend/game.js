import {
  DrawingUtils,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import {
  classifyFace,
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
  drawPlayerOutlines,
  drawPoses,
} from "./pose-tracking.js";

// Set model limits, frame cadence, track lifetime, and per-cue repeat cooldown.
const MAX_FACES = 5;
const MAX_HANDS = 4;
const MAX_POSES = 5;
const INFERENCE_INTERVAL_MS = 1000 / 12;
const TRACK_TIMEOUT_MS = 3500;
const CUE_COOLDOWN_MS = 1200;
const PLAYER_TIMER_MS = 2 * 60 * 1000;
const STORAGE_KEY = "definitly-not-glorified-surveillance-system-v1";
const LEGACY_STORAGE_KEY = "cue-carnival-game-v1";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const OBJECTRON_QUERY_INTERVAL_MS = 250;
const OBJECT_TRACK_TIMEOUT_MS = 2500;

// Connect each detectable cue to its slider, display label, and default points.
const CUES = [
  { id: "fist", label: "Fist", input: "points-fist" },
  { id: "raised-fist", label: "Raised fist", input: "points-raised-fist" },
  { id: "middle-finger", label: "Middle finger", input: "points-middle-finger" },
  { id: "open-mouth", label: "Open mouth", input: "points-open-mouth" },
  { id: "lowered-brows", label: "Lowered brows", input: "points-lowered-brows" },
  { id: "both-arms-raised", label: "Both arms raised", input: "points-both-arms-raised" },
  { id: "arms-spread", label: "Arms spread", input: "points-arms-spread" },
];
const OBJECT_BONUSES = [
  { id: "shoe", modelName: "Shoe", label: "Shoe", input: "seconds-shoe" },
  { id: "chair", modelName: "Chair", label: "Chair", input: "seconds-chair" },
  { id: "cup", modelName: "Cup", label: "Cup", input: "seconds-cup" },
  { id: "camera", modelName: "Camera", label: "Camera", input: "seconds-camera" },
];
const cueById = new Map(CUES.map((cue) => [cue.id, cue]));

// Load only saved game data and tolerate missing, invalid, or blocked storage.
function readSavedGame() {
  try {
    const savedValue = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY);
    const saved = JSON.parse(savedValue ?? "null");
    const storedFaceScores = Array.isArray(saved?.faceScores)
      ? saved.faceScores.filter((score) => Number.isInteger(score.id) && Number.isFinite(score.points))
      : Array.isArray(saved?.players)
        ? saved.players.filter((score) => Number.isInteger(score.id) && Number.isFinite(score.points))
      : [];
    const faceScores = storedFaceScores.map((score) => {
      const previousAction = typeof score.lastEvent === "string"
        ? score.lastEvent.replace(/\s+\+\d+(?:\.\d+)?$/, "")
        : "";
      const actions = {};
      if (Array.isArray(score.actions)) {
        for (const action of score.actions) {
          const cue = CUES.find(({ label }) => label === action);
          if (cue) actions[cue.id] = 1;
        }
      } else if (score.actions && typeof score.actions === "object") {
        for (const cue of CUES) {
          const count = score.actions[cue.id];
          if (Number.isInteger(count) && count > 0) actions[cue.id] = count;
        }
      } else if (previousAction && !["Ready for cues", "No recent cue", "Points cleared"].includes(previousAction)) {
        const cue = CUES.find(({ label }) => label === previousAction);
        if (cue) actions[cue.id] = 1;
      }
      return { id: score.id, points: score.points, actions };
    });
    const unassignedScores = Array.isArray(saved?.unassignedScores)
      ? saved.unassignedScores.filter(Number.isFinite)
      : [];
    const highestId = faceScores.reduce((highest, score) => Math.max(highest, score.id), 0);
    return {
      nextFaceId: Math.max(highestId + 1, Number(saved?.nextFaceId) || 1),
      faceScores,
      unassignedScores,
      settings: saved?.settings && typeof saved.settings === "object" ? saved.settings : {},
    };
  } catch {
    return { nextFaceId: 1, faceScores: [], unassignedScores: [], settings: {} };
  }
}

const savedGame = readSavedGame();
const faceScores = new Map(savedGame.faceScores.map((score) => [score.id, score]));
let unassignedScores = savedGame.unassignedScores;
// Keep active camera and face-track state in memory; only the scoreboard is persisted.
let nextFaceId = savedGame.nextFaceId;
let storageEnabled = true;
let faceLandmarker;
let handLandmarker;
let poseLandmarker;
let cameraStream;
let animationFrameId;
let lastInferenceAt = 0;
let lastObjectronInferenceAt = 0;
let objectronIndex = 0;
let objectronInferenceInFlight = false;
let objectronDetectors = [];
const faceTracks = new Map();
const objectTracks = new Map(OBJECT_BONUSES.map(({ id }) => [id, []]));

// Cache the controls and output elements used by the game page.
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const deleteSavedDataButton = document.querySelector("#delete-saved-data-button");
const statusElement = document.querySelector("#status");
const video = document.querySelector("#camera");
const overlay = document.querySelector("#tracking-overlay");
const overlayContext = overlay.getContext("2d");
const criminalAlert = document.querySelector("#criminal-alert");
const faceCount = document.querySelector("#face-count");
const handCount = document.querySelector("#hand-count");
const poseCount = document.querySelector("#pose-count");
const scoreList = document.querySelector("#score-list");
const drawingUtils = new DrawingUtils(overlayContext);

// Show camera, permission, and storage feedback in the page's live status region.
function setStatus(message) {
  if (statusElement) statusElement.textContent = message;
}

// Persist the scoreboard and cue settings unless the user deleted saved data this session.
function saveGame() {
  if (!storageEnabled) return;

  try {
    const savedFaceScores = [...faceScores.values()];
    const savedData = { nextFaceId, faceScores: savedFaceScores, unassignedScores };
    savedData.settings = Object.fromEntries([
      ...CUES.map(({ id, input }) => [id, Number(document.querySelector(`#${input}`).value)]),
      ...OBJECT_BONUSES.map(({ id, input }) => [`object-${id}`, Number(document.querySelector(`#${input}`).value)]),
    ]);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(savedData));
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  } catch {
    setStatus("Browser storage is unavailable; this session's scores will not persist.");
  }
}

// Read the current slider value for a detected cue.
function getPoints(cueId) {
  const cue = cueById.get(cueId);
  return cue ? Number(document.querySelector(`#${cue.input}`).value) : 0;
}

function getObjectBonusSeconds(objectId) {
  const object = OBJECT_BONUSES.find(({ id }) => id === objectId);
  return object ? Number(document.querySelector(`#${object.input}`).value) : 0;
}

// Create a zero-point score and empty action list for a newly numbered face.
function ensureFaceScore(id) {
  if (!faceScores.has(id)) {
    faceScores.set(id, { id, points: 0, actions: [] });
  }
  return faceScores.get(id);
}

// Render each face's number, point total, and count of each detected action.
function updateScoreboard() {
  scoreList.replaceChildren();
  const standings = [
    ...[...faceScores.values()].map((score) => ({ ...score, kind: "face" })),
    ...unassignedScores.map((points, index) => ({ points, index, kind: "unassigned" })),
  ].sort((first, second) =>
    second.points - first.points || (first.id ?? Infinity) - (second.id ?? Infinity),
  );

  if (standings.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "No points yet. Start the camera to begin.";
    scoreList.append(empty);
    return;
  }

  for (const [rank, scoreEntry] of standings.entries()) {
    const item = document.createElement("li");
    const actions = scoreEntry.kind === "face"
      ? Object.entries(scoreEntry.actions)
        .filter(([cueId, count]) => cueById.has(cueId) && count > 0)
        .map(([cueId, count]) => `${cueById.get(cueId).label} (${count})`)
        .join(", ") || "None"
      : "Unavailable (face number cleared)";
    item.textContent = scoreEntry.kind === "face"
      ? `Face (${scoreEntry.id}), Points: ${scoreEntry.points}, List of actions: ${actions}`
      : `Unassigned total (${scoreEntry.index + 1}), Points: ${scoreEntry.points}, List of actions: ${actions}`;
    scoreList.append(item);
  }
}

// Mirror slider values in their outputs and optionally persist the settings.
function updatePointControls(save = true) {
  for (const cue of CUES) {
    const input = document.querySelector(`#${cue.input}`);
    const output = document.querySelector(`#${cue.input}-value`);
    output.value = `+${input.value}`;
  }
  for (const object of OBJECT_BONUSES) {
    const input = document.querySelector(`#${object.input}`);
    const output = document.querySelector(`#${object.input}-value`);
    output.value = `+${input.value}s`;
  }
  if (save) saveGame();
}

// Find the lowest-cost one-to-one column for every detection using the Hungarian method.
function assignMinimumCostPairs(costMatrix) {
  const rowCount = costMatrix.length;
  if (rowCount === 0) return [];

  const columnCount = costMatrix[0].length;
  const rowPotentials = new Array(rowCount + 1).fill(0);
  const columnPotentials = new Array(columnCount + 1).fill(0);
  const columnOwners = new Array(columnCount + 1).fill(0);
  const previousColumns = new Array(columnCount + 1).fill(0);

  for (let row = 1; row <= rowCount; row += 1) {
    columnOwners[0] = row;
    let currentColumn = 0;
    const minimumCosts = new Array(columnCount + 1).fill(Infinity);
    const usedColumns = new Array(columnCount + 1).fill(false);

    do {
      usedColumns[currentColumn] = true;
      const currentRow = columnOwners[currentColumn];
      let nextColumn = 0;
      let costDelta = Infinity;

      for (let column = 1; column <= columnCount; column += 1) {
        if (usedColumns[column]) continue;
        const reducedCost = costMatrix[currentRow - 1][column - 1]
          - rowPotentials[currentRow]
          - columnPotentials[column];
        if (reducedCost < minimumCosts[column]) {
          minimumCosts[column] = reducedCost;
          previousColumns[column] = currentColumn;
        }
        if (minimumCosts[column] < costDelta) {
          costDelta = minimumCosts[column];
          nextColumn = column;
        }
      }

      for (let column = 0; column <= columnCount; column += 1) {
        if (usedColumns[column]) {
          rowPotentials[columnOwners[column]] += costDelta;
          columnPotentials[column] -= costDelta;
        } else {
          minimumCosts[column] -= costDelta;
        }
      }
      currentColumn = nextColumn;
    } while (columnOwners[currentColumn] !== 0);

    do {
      const previousColumn = previousColumns[currentColumn];
      columnOwners[currentColumn] = columnOwners[previousColumn];
      currentColumn = previousColumn;
    } while (currentColumn !== 0);
  }

  const assignments = new Array(rowCount).fill(-1);
  for (let column = 1; column <= columnCount; column += 1) {
    const row = columnOwners[column];
    if (row !== 0) assignments[row - 1] = column - 1;
  }
  return assignments;
}

// Match detections to predicted tracks using a global one-to-one set of candidate pairs.
function matchFaces(faceLandmarks, now) {
  for (const [id, track] of faceTracks) {
    if (now - track.lastSeen >= TRACK_TIMEOUT_MS) faceTracks.delete(id);
  }
  const detections = faceLandmarks.map(getFaceBounds);
  const tracks = [...faceTracks.values()];
  const unmatchedCost = 1.05;
  const invalidCost = 100;
  const costMatrix = detections.map((bounds) => {
    const trackCosts = tracks.map((track) => {
      const elapsed = Math.min((now - track.lastSeen) / 1000, 0.75);
      const predictedX = track.centerX + track.velocityX * elapsed;
      const predictedY = track.centerY + track.velocityY * elapsed;
      const distance = Math.hypot(bounds.centerX - predictedX, bounds.centerY - predictedY);
      const speed = Math.hypot(track.velocityX, track.velocityY);
      const distanceLimit = Math.max(0.08, (bounds.width + track.width) * 0.6)
        + Math.min(0.12, speed * elapsed * 0.4);
      if (distance > distanceLimit) return invalidCost;

      const widthChange = Math.abs(Math.log((bounds.width + 0.01) / (track.width + 0.01)));
      const heightChange = Math.abs(Math.log((bounds.height + 0.01) / (track.height + 0.01)));
      return (distance / distanceLimit) * 0.75
        + Math.min(1, (widthChange + heightChange) * 0.4) * 0.25;
    });
    return [...trackCosts, ...new Array(detections.length).fill(unmatchedCost)];
  });
  const assignments = assignMinimumCostPairs(costMatrix);

  return detections.map((bounds, detectionIndex) => {
    const trackIndex = assignments[detectionIndex];
    const track = trackIndex >= 0 && trackIndex < tracks.length ? tracks[trackIndex] : undefined;
    if (track) {
      const elapsed = Math.max((now - track.lastSeen) / 1000, INFERENCE_INTERVAL_MS / 1000);
      const measuredVelocityX = (bounds.centerX - track.centerX) / elapsed;
      const measuredVelocityY = (bounds.centerY - track.centerY) / elapsed;
      track.velocityX = track.velocityX * 0.6 + Math.max(-1.5, Math.min(1.5, measuredVelocityX)) * 0.4;
      track.velocityY = track.velocityY * 0.6 + Math.max(-1.5, Math.min(1.5, measuredVelocityY)) * 0.4;
      track.centerX = bounds.centerX;
      track.centerY = bounds.centerY;
      track.width = bounds.width;
      track.height = bounds.height;
      track.lastSeen = now;
      return track;
    }

    const newTrack = {
      id: nextFaceId++,
      centerX: bounds.centerX,
      centerY: bounds.centerY,
      width: bounds.width,
      height: bounds.height,
      velocityX: 0,
      velocityY: 0,
      lastSeen: now,
      timerUpdatedAt: now,
      remainingMs: PLAYER_TIMER_MS,
      elapsedMs: 0,
      cueStates: new Map(),
      lastAwardAt: new Map(),
    };
    faceTracks.set(newTrack.id, newTrack);
    ensureFaceScore(newTrack.id);
    saveGame();
    return newTrack;
  });
}

// Associate a pose head or hand point with the nearest visible face, if close enough.
function closestFaceId(point, faceLandmarks, tracks, maxDistance) {
  if (!point) return undefined;
  let closestId;
  let closestDistance = maxDistance;

  for (let index = 0; index < faceLandmarks.length; index += 1) {
    const bounds = getFaceBounds(faceLandmarks[index]);
    const distance = Math.hypot(point.x - bounds.centerX, point.y - bounds.centerY);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestId = tracks[index].id;
    }
  }
  return closestId;
}

// Estimate the head location from the pose model's face and eye landmarks.
function poseHead(pose) {
  const headPoints = pose.slice(0, 7).filter((point) => point && (point.visibility ?? 1) >= 0.4);
  if (headPoints.length === 0) return undefined;
  return {
    x: headPoints.reduce((total, point) => total + point.x, 0) / headPoints.length,
    y: headPoints.reduce((total, point) => total + point.y, 0) / headPoints.length,
  };
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.ceil(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function updatePlayerTimers(tracks, now) {
  for (const track of tracks) {
    const elapsed = Math.max(0, now - track.timerUpdatedAt);
    track.elapsedMs += elapsed;
    track.remainingMs = Math.max(0, track.remainingMs - elapsed);
    track.timerUpdatedAt = now;
  }
}

function updateCriminalAlert(tracks) {
  const recognizedIds = tracks.filter((track) => track.remainingMs === 0).map((track) => track.id);
  criminalAlert.hidden = recognizedIds.length === 0;
  criminalAlert.textContent = recognizedIds.length
    ? `CRIMINAL RECOGNIZED - FACE ${recognizedIds.join(", FACE ")}`
    : "";
}

function getObjectBounds(detectedObject) {
  const points = detectedObject.keypoints
    ?.map(({ point2d }) => point2d)
    .filter((point) => point && Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!points?.length) return undefined;

  const left = Math.min(...points.map(({ x }) => x));
  const right = Math.max(...points.map(({ x }) => x));
  const top = Math.min(...points.map(({ y }) => y));
  const bottom = Math.max(...points.map(({ y }) => y));
  return {
    left,
    top,
    width: right - left,
    height: bottom - top,
    centerX: (left + right) / 2,
    centerY: (top + bottom) / 2,
  };
}

function getObjectOwner(bounds, now) {
  const nearest = [...faceTracks.values()]
    .filter((track) => now - track.lastSeen < TRACK_TIMEOUT_MS)
    .map((track) => ({
      track,
      distance: Math.hypot(track.centerX - bounds.centerX, (track.centerY - bounds.centerY) * 0.5),
    }))
    .sort((first, second) => first.distance - second.distance)[0];
  return nearest && nearest.distance < 0.55 ? nearest.track : undefined;
}

function handleObjectronResults(objectId, results) {
  if (!cameraStream) return;
  const now = performance.now();
  const existingTracks = (objectTracks.get(objectId) ?? [])
    .filter((track) => now - track.lastSeen < OBJECT_TRACK_TIMEOUT_MS);
  const unmatchedTracks = new Set(existingTracks);

  for (const detection of results.objectDetections ?? []) {
    const bounds = getObjectBounds(detection);
    if (!bounds) continue;
    let closest;
    let closestDistance = Infinity;
    for (const track of unmatchedTracks) {
      const distance = Math.hypot(track.bounds.centerX - bounds.centerX, track.bounds.centerY - bounds.centerY);
      const matchDistance = Math.max(0.12, Math.hypot(track.bounds.width, track.bounds.height) * 0.35);
      if (distance < matchDistance && distance < closestDistance) {
        closest = track;
        closestDistance = distance;
      }
    }

    if (closest) {
      closest.bounds = bounds;
      closest.lastSeen = now;
      unmatchedTracks.delete(closest);
      continue;
    }

    const objectTrack = { bounds, lastSeen: now };
    existingTracks.push(objectTrack);
    const owner = getObjectOwner(bounds, now);
    const bonusSeconds = getObjectBonusSeconds(objectId);
    if (owner && owner.remainingMs > 0 && bonusSeconds > 0) {
      owner.remainingMs += bonusSeconds * 1000;
    }
  }

  objectTracks.set(objectId, existingTracks);
}

function drawObjectronDetections(context, canvas) {
  context.save();
  context.lineWidth = Math.max(2, canvas.width / 400);
  for (const object of OBJECT_BONUSES) {
    const tracks = objectTracks.get(object.id) ?? [];
    for (const track of tracks) {
      const { left, top, width, height } = track.bounds;
      context.strokeStyle = "#55e0c0";
      context.strokeRect(left * canvas.width, top * canvas.height, width * canvas.width, height * canvas.height);
      context.font = `700 ${Math.max(12, canvas.width / 48)}px monospace`;
      context.fillStyle = "#55e0c0";
      context.fillText(object.label.toUpperCase(), left * canvas.width, Math.max(14, top * canvas.height - 4));
    }
  }
  context.restore();
}

async function initializeObjectron() {
  if (!window.Objectron) throw new Error("MediaPipe Objectron failed to load.");
  return Promise.all(OBJECT_BONUSES.map(async (object) => {
    const detector = new window.Objectron({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/objectron/${file}`,
    });
    await detector.setOptions({
      modelName: object.modelName,
      maxNumObjects: 3,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.75,
    });
    detector.onResults((results) => handleObjectronResults(object.id, results));
    return detector;
  }));
}

async function runObjectronFrame(now) {
  if (
    objectronDetectors.length === 0 ||
    objectronInferenceInFlight ||
    now - lastObjectronInferenceAt < OBJECTRON_QUERY_INTERVAL_MS
  ) return;

  const detector = objectronDetectors[objectronIndex];
  objectronIndex = (objectronIndex + 1) % objectronDetectors.length;
  lastObjectronInferenceAt = now;
  objectronInferenceInFlight = true;
  try {
    await detector.send({ image: video });
  } catch (error) {
    setStatus(`Objectron stopped: ${error.message}`);
  } finally {
    objectronInferenceInFlight = false;
  }
}

// Combine facial expressions, body poses, and hand gestures under a face track.
function collectCueEvents(faceResult, handResult, poseResult, tracks) {
  const cuesByFace = new Map(tracks.map((track) => [track.id, new Set()]));
  const addCue = (id, cue) => cuesByFace.get(id)?.add(cue);

  for (const [index, track] of tracks.entries()) {
    const blendshapes = faceResult.faceBlendshapes[index];
    if (blendshapes) {
      for (const cue of classifyFace(blendshapes)) addCue(track.id, cue);
    }
  }

  const poseOwners = poseResult.landmarks.map((pose) => {
    const id = closestFaceId(poseHead(pose), faceResult.faceLandmarks, tracks, 0.3);
    if (id !== undefined) {
      for (const cue of classifyPose(pose)) addCue(id, cue);
      return { id, landmarks: pose };
    }
    return undefined;
  });

  for (const hand of handResult.landmarks) {
    let cue = classifyGesture(hand);
    if (!cue) continue;

    const wrist = hand[0];
    let owner;
    let ownerPose;
    let ownerPoseWristIndex;
    let nearestWristDistance = 0.28;
    for (const [poseIndex, pose] of poseResult.landmarks.entries()) {
      const poseOwner = poseOwners[poseIndex];
      if (!poseOwner) continue;
      for (const wristIndex of [15, 16]) {
        const poseWrist = pose[wristIndex];
        if (!poseWrist || (poseWrist.visibility ?? 1) < 0.35) continue;
        const distance = Math.hypot(wrist.x - poseWrist.x, wrist.y - poseWrist.y);
        if (distance < nearestWristDistance) {
          nearestWristDistance = distance;
          owner = poseOwner.id;
          ownerPose = pose;
          ownerPoseWristIndex = wristIndex;
        }
      }
    }

    if (cue === "fist" && ownerPose) {
      const shoulder = ownerPose[ownerPoseWristIndex === 15 ? 11 : 12];
      if (shoulder && (shoulder.visibility ?? 1) >= 0.35 && wrist.y < shoulder.y - 0.06) {
        cue = "raised-fist";
      }
    }

    owner ??= closestFaceId(wrist, faceResult.faceLandmarks, tracks, 0.32);
    if (owner !== undefined) addCue(owner, cue);
  }

  return cuesByFace;
}

// Require two detected frames to activate and two missing frames to end a cue.
function scoreCueEvents(cuesByFace, now) {
  let changed = false;
  for (const track of faceTracks.values()) {
    const detectedCues = cuesByFace.get(track.id) ?? new Set();
    const faceScore = ensureFaceScore(track.id);
    const cueIds = new Set([...track.cueStates.keys(), ...detectedCues]);

    for (const cueId of cueIds) {
      const cue = cueById.get(cueId);
      if (!cue) continue;
      const state = track.cueStates.get(cueId) ?? {
        detectedFrames: 0,
        missingFrames: 0,
        active: false,
      };

      if (detectedCues.has(cueId)) {
        state.detectedFrames += 1;
        state.missingFrames = 0;
        if (!state.active && state.detectedFrames >= 2) {
          state.active = true;
          faceScore.actions[cueId] = (faceScore.actions[cueId] ?? 0) + 1;
          changed = true;

          const lastAwardAt = track.lastAwardAt.get(cueId) ?? -Infinity;
          if (now - lastAwardAt >= CUE_COOLDOWN_MS) {
            const cuePoints = getPoints(cueId);
            faceScore.points += cuePoints;
            track.remainingMs = Math.max(0, track.remainingMs - cuePoints * 1000);
            track.lastAwardAt.set(cueId, now);
            changed = true;
          }
        }
      } else {
        state.detectedFrames = 0;
        state.missingFrames += 1;
        if (state.missingFrames >= 2) state.active = false;
      }

      track.cueStates.set(cueId, state);
    }
  }

  if (changed) saveGame();
}

// Run throttled model inference, update counts and scores, and redraw the overlay.
function detectFrame() {
  if (!cameraStream) return;
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
    const tracks = matchFaces(faceResult.faceLandmarks, now);
    const cuesByFace = collectCueEvents(faceResult, handResult, poseResult, tracks);
    runObjectronFrame(now);
    updatePlayerTimers(tracks, now);

    faceCount.value = String(faceResult.faceLandmarks.length);
    handCount.value = String(handResult.landmarks.length);
    poseCount.value = String(poseResult.landmarks.length);
    scoreCueEvents(cuesByFace, now);

    drawFaceMesh(faceResult, video, overlay, overlayContext, drawingUtils, tracks.map((track) => ({
      ...track,
      remainingLabel: formatDuration(track.remainingMs),
      elapsedLabel: formatDuration(track.elapsedMs),
    })));
    drawPoses(poseResult, drawingUtils);
    drawHands(handResult, drawingUtils);
    const tracksByPose = poseResult.landmarks.map((pose) => {
      const ownerId = closestFaceId(poseHead(pose), faceResult.faceLandmarks, tracks, 0.3);
      const track = tracks.find(({ id }) => id === ownerId);
      return track ? {
        ...track,
        progress: 1 - track.remainingMs / PLAYER_TIMER_MS,
      } : undefined;
    });
    drawPlayerOutlines(poseResult, overlay, overlayContext, tracksByPose);
    drawObjectronDetections(overlayContext, overlay);
    updateCriminalAlert(tracks);
    updateScoreboard();
  } catch (error) {
    stopCamera(`Tracking stopped: ${error.message}`);
    return;
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

// Request camera permission, initialize all three models, and start the frame loop.
async function startCamera() {
  startButton.disabled = true;
  setStatus("Waiting for camera permission…");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access needs HTTPS and a supported browser.");
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
      setStatus("Loading face, hand, and pose models…");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      [faceLandmarker, handLandmarker, poseLandmarker] = await Promise.all([
        faceLandmarker ?? createFaceLandmarker(vision, MAX_FACES),
        handLandmarker ?? createHandLandmarker(vision, MAX_HANDS),
        poseLandmarker ?? createPoseLandmarker(vision, MAX_POSES),
      ]);
    }

    if (objectronDetectors.length === 0) {
      setStatus("Loading MediaPipe Objectron models…");
      objectronDetectors = await initializeObjectron();
    }

    stopButton.disabled = false;
    lastInferenceAt = 0;
    lastObjectronInferenceAt = 0;
    setStatus("Camera live. Objectron detects shoes, chairs, cups and cameras. Settings stay on this device.");
    animationFrameId = requestAnimationFrame(detectFrame);
  } catch (error) {
    stopCamera(`Could not start camera or models: ${error.message}`);
  } finally {
    startButton.disabled = false;
  }
}

// Stop the camera, clear transient tracks, and leave saved scoreboard data intact.
function stopCamera(message = "Camera off. Scores stay saved on this device.") {
  if (animationFrameId !== undefined) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = undefined;
  }
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  criminalAlert.hidden = true;
  criminalAlert.textContent = "";
  faceTracks.clear();
  for (const objectId of objectTracks.keys()) objectTracks.set(objectId, []);
  faceCount.value = "0";
  handCount.value = "0";
  poseCount.value = "0";
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus(message);
  updateScoreboard();
}

// Delete stored game data and clear the session so the old scoreboard cannot reappear.
function deleteSavedData() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(LEGACY_STORAGE_KEY);
    storageEnabled = false;
    stopCamera("Saved data deleted. Camera stopped.");
    faceScores.clear();
    unassignedScores = [];
    nextFaceId = 1;
    faceTracks.clear();
    updateScoreboard();
    setStatus("Saved data deleted. Scores are cleared for this session.");
  } catch {
    setStatus("Could not delete saved data from browser storage.");
  }
}

// Restore saved slider choices when present; defaults remain unsaved on initial load.
for (const cue of CUES) {
  const input = document.querySelector(`#${cue.input}`);
  const savedValue = savedGame.settings[cue.id];
  if (Number.isFinite(savedValue)) input.value = String(Math.max(0, Math.min(10, savedValue)));
  input.addEventListener("input", updatePointControls);
}

for (const object of OBJECT_BONUSES) {
  const input = document.querySelector(`#${object.input}`);
  const savedValue = savedGame.settings[`object-${object.id}`];
  if (Number.isFinite(savedValue)) input.value = String(Math.max(0, Math.min(60, savedValue)));
  input.addEventListener("input", updatePointControls);
}

// Connect page actions and initialize the scoreboard without rewriting cleared settings.
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", () => stopCamera());
deleteSavedDataButton.addEventListener("click", deleteSavedData);
window.addEventListener("pagehide", () => stopCamera("Camera closed."));

updatePointControls(false);
updateScoreboard();
