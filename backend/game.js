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
  drawPoses,
} from "./pose-tracking.js";

// Set model limits, frame cadence, track lifetime, and per-cue repeat cooldown.
const MAX_FACES = 5;
const MAX_HANDS = 4;
const MAX_POSES = 5;
const INFERENCE_INTERVAL_MS = 1000 / 12;
const TRACK_TIMEOUT_MS = 3500;
const CUE_COOLDOWN_MS = 1200;
const STORAGE_KEY = "definitly-not-glorified-surveillance-system-v1";
const LEGACY_STORAGE_KEY = "cue-carnival-game-v1";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

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
const cueById = new Map(CUES.map((cue) => [cue.id, cue]));

// Load only saved game data and tolerate missing, invalid, or blocked storage.
function readSavedGame() {
  try {
    const savedValue = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY);
    const saved = JSON.parse(savedValue ?? "null");
    const storedPlayerScores = Array.isArray(saved?.playerScores)
      ? saved.playerScores.filter((score) => Number.isInteger(score.id) && Number.isFinite(score.points))
      : Array.isArray(saved?.faceScores)
        ? saved.faceScores.filter((score) => Number.isInteger(score.id) && Number.isFinite(score.points))
      : Array.isArray(saved?.players)
        ? saved.players.filter((score) => Number.isInteger(score.id) && Number.isFinite(score.points))
      : [];
    const playerScores = storedPlayerScores.map((score) => {
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
    const highestId = playerScores.reduce((highest, score) => Math.max(highest, score.id), 0);
    return {
      nextPlayerId: Math.max(highestId + 1, Number(saved?.nextPlayerId) || Number(saved?.nextFaceId) || 1),
      playerScores,
      unassignedScores,
      settings: saved?.settings && typeof saved.settings === "object" ? saved.settings : {},
    };
  } catch {
    return { nextPlayerId: 1, playerScores: [], unassignedScores: [], settings: {} };
  }
}

const savedGame = readSavedGame();
const playerScores = new Map(savedGame.playerScores.map((score) => [score.id, score]));
let unassignedScores = savedGame.unassignedScores;
// Keep active pose tracks in memory; only the scoreboard is persisted.
let nextPlayerId = savedGame.nextPlayerId;
let storageEnabled = true;
let faceLandmarker;
let handLandmarker;
let poseLandmarker;
let cameraStream;
let animationFrameId;
let lastInferenceAt = 0;
const poseTracks = new Map();

// Cache the controls and output elements used by the game page.
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const deleteSavedDataButton = document.querySelector("#delete-saved-data-button");
const statusElement = document.querySelector("#status");
const video = document.querySelector("#camera");
const overlay = document.querySelector("#tracking-overlay");
const overlayContext = overlay.getContext("2d");
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
    const savedPlayerScores = [...playerScores.values()];
    const savedData = { nextPlayerId, playerScores: savedPlayerScores, unassignedScores };
    savedData.settings = Object.fromEntries(
      CUES.map(({ id, input }) => [id, Number(document.querySelector(`#${input}`).value)]),
    );
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

// Create a zero-point score and empty action list for a newly numbered player.
function ensurePlayerScore(id) {
  if (!playerScores.has(id)) {
    playerScores.set(id, { id, points: 0, actions: [] });
  }
  return playerScores.get(id);
}

// Render each player's number, point total, and count of each detected action.
function updateScoreboard() {
  scoreList.replaceChildren();
  const standings = [
    ...[...playerScores.values()].map((score) => ({ ...score, kind: "player" })),
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
    const actions = scoreEntry.kind === "player"
      ? Object.entries(scoreEntry.actions)
        .filter(([cueId, count]) => cueById.has(cueId) && count > 0)
        .map(([cueId, count]) => `${cueById.get(cueId).label} (${count})`)
        .join(", ") || "None"
      : "Unavailable (player number cleared)";
    item.textContent = scoreEntry.kind === "player"
      ? `Player (${scoreEntry.id}), Points: ${scoreEntry.points}, List of actions: ${actions}`
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

// Track people by torso landmarks so player IDs survive face detection changes.
function matchPoses(poseLandmarks, now) {
  for (const [id, track] of poseTracks) {
    if (now - track.lastSeen >= TRACK_TIMEOUT_MS) poseTracks.delete(id);
  }

  const detections = poseLandmarks.map((pose) => {
    const torso = [11, 12, 23, 24].map((index) => pose[index])
      .filter((point) => point && (point.visibility ?? 1) >= 0.35);
    if (torso.length < 2) return undefined;
    const anchors = Object.fromEntries([0, 11, 12, 23, 24].flatMap((pointIndex) => {
      const point = pose[pointIndex];
      return point && (point.visibility ?? 1) >= 0.4
        ? [[pointIndex, { x: point.x, y: point.y }]]
        : [];
    }));
    const centerX = torso.reduce((total, point) => total + point.x, 0) / torso.length;
    const centerY = torso.reduce((total, point) => total + point.y, 0) / torso.length;
    const width = Math.max(...torso.map(({ x }) => x)) - Math.min(...torso.map(({ x }) => x));
    const height = Math.max(...torso.map(({ y }) => y)) - Math.min(...torso.map(({ y }) => y));
    return { centerX, centerY, width, height, anchors };
  });
  const tracks = [...poseTracks.values()];
  const unmatchedCost = 1.05;
  const invalidCost = 100;
  const costMatrix = detections.map((bounds) => {
    if (!bounds) return [...tracks.map(() => invalidCost), ...new Array(detections.length).fill(unmatchedCost)];
    const trackCosts = tracks.map((track) => {
      const elapsed = Math.min((now - track.lastSeen) / 1000, 0.75);
      const predictedX = track.centerX + track.velocityX * elapsed;
      const predictedY = track.centerY + track.velocityY * elapsed;
      const centerDistance = Math.hypot(bounds.centerX - predictedX, bounds.centerY - predictedY);
      const movement = Math.hypot(track.velocityX, track.velocityY) * elapsed;
      const centerLimit = Math.max(0.18, (bounds.width + track.width) * 1.1 + movement * 0.5);
      const anchorDistances = Object.entries(bounds.anchors).flatMap(([key, point]) => {
        const previous = track.anchors[key];
        if (!previous) return [];
        const anchorX = previous.x + previous.velocityX * elapsed;
        const anchorY = previous.y + previous.velocityY * elapsed;
        return [Math.hypot(point.x - anchorX, point.y - anchorY)];
      });
      const skeletonDistance = anchorDistances.length > 0
        ? anchorDistances.reduce((total, distance) => total + distance, 0) / anchorDistances.length
        : centerDistance;
      const skeletonLimit = Math.max(0.18, Math.hypot(bounds.width + track.width, bounds.height + track.height) * 0.75 + movement * 0.5);
      if (
        centerDistance > centerLimit ||
        (anchorDistances.length >= 2 && skeletonDistance > skeletonLimit)
      ) return invalidCost;

      const widthChange = Math.abs(Math.log((bounds.width + 0.03) / (track.width + 0.03)));
      const heightChange = Math.abs(Math.log((bounds.height + 0.03) / (track.height + 0.03)));
      const shapeChange = Math.min(1, (widthChange + heightChange) * 0.5);
      return centerDistance / centerLimit * 0.25
        + skeletonDistance / skeletonLimit * 0.6
        + shapeChange * 0.15;
    });
    return [...trackCosts, ...new Array(detections.length).fill(unmatchedCost)];
  });
  const assignments = assignMinimumCostPairs(costMatrix);

  return detections.map((bounds, index) => {
    if (!bounds) return undefined;
    const trackIndex = assignments[index];
    const track = trackIndex >= 0 && trackIndex < tracks.length ? tracks[trackIndex] : undefined;
    if (track) {
      const elapsed = Math.max((now - track.lastSeen) / 1000, INFERENCE_INTERVAL_MS / 1000);
      const measuredVelocityX = (bounds.centerX - track.centerX) / elapsed;
      const measuredVelocityY = (bounds.centerY - track.centerY) / elapsed;
      track.velocityX = track.velocityX * 0.6 + Math.max(-1.5, Math.min(1.5, measuredVelocityX)) * 0.4;
      track.velocityY = track.velocityY * 0.6 + Math.max(-1.5, Math.min(1.5, measuredVelocityY)) * 0.4;
      const anchors = {};
      for (const [key, point] of Object.entries(bounds.anchors)) {
        const previous = track.anchors[key];
        const velocityX = previous ? (point.x - previous.x) / elapsed : 0;
        const velocityY = previous ? (point.y - previous.y) / elapsed : 0;
        anchors[key] = {
          x: point.x,
          y: point.y,
          velocityX: previous
            ? previous.velocityX * 0.6 + Math.max(-1.5, Math.min(1.5, velocityX)) * 0.4
            : 0,
          velocityY: previous
            ? previous.velocityY * 0.6 + Math.max(-1.5, Math.min(1.5, velocityY)) * 0.4
            : 0,
        };
      }
      Object.assign(track, bounds, { anchors, lastSeen: now });
      return track;
    }

    const newTrack = {
      id: nextPlayerId++,
      ...bounds,
      anchors: Object.fromEntries(Object.entries(bounds.anchors).map(([key, point]) => [key, {
        ...point,
        velocityX: 0,
        velocityY: 0,
      }])),
      velocityX: 0,
      velocityY: 0,
      lastSeen: now,
      cueStates: new Map(),
      lastAwardAt: new Map(),
    };
    poseTracks.set(newTrack.id, newTrack);
    ensurePlayerScore(newTrack.id);
    saveGame();
    return newTrack;
  });
}

function matchFacesToPlayers(faceLandmarks, poseLandmarks, tracks) {
  const maxDistance = 0.3;
  const invalidCost = 100;
  const costMatrix = faceLandmarks.map((face) => {
    const bounds = getFaceBounds(face);
    const poseCosts = poseLandmarks.map((pose, index) => {
      const head = poseHead(pose);
      if (!head || !tracks[index]) return invalidCost;
      const distance = Math.hypot(bounds.centerX - head.x, bounds.centerY - head.y);
      return distance < maxDistance ? distance : invalidCost;
    });
    return [...poseCosts, ...new Array(faceLandmarks.length).fill(maxDistance)];
  });
  const assignments = assignMinimumCostPairs(costMatrix);
  return assignments.map((poseIndex, faceIndex) =>
    poseIndex >= 0 && poseIndex < tracks.length && costMatrix[faceIndex][poseIndex] < maxDistance
      ? tracks[poseIndex].id
      : undefined,
  );
}

function closestMappedPlayerId(point, faceLandmarks, facePlayerIds, maxDistance) {
  if (!point) return undefined;
  let closestId;
  let closestDistance = maxDistance;
  for (const [index, face] of faceLandmarks.entries()) {
    const bounds = getFaceBounds(face);
    const distance = Math.hypot(point.x - bounds.centerX, point.y - bounds.centerY);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestId = facePlayerIds[index];
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

// Combine facial expressions, body poses, and hand gestures under a pose track.
function collectCueEvents(faceResult, handResult, poseResult, facePlayerIds, playerTracks) {
  const cuesByPlayer = new Map(playerTracks.filter(Boolean).map((track) => [track.id, new Set()]));
  const addCue = (id, cue) => cuesByPlayer.get(id)?.add(cue);

  for (const [index, playerId] of facePlayerIds.entries()) {
    if (playerId === undefined) continue;
    const blendshapes = faceResult.faceBlendshapes[index];
    if (blendshapes) {
      for (const cue of classifyFace(blendshapes)) addCue(playerId, cue);
    }
  }

  const poseOwners = poseResult.landmarks.map((pose, index) => {
    const track = playerTracks[index];
    if (!track) return undefined;
    for (const cue of classifyPose(pose)) addCue(track.id, cue);
    return { id: track.id, landmarks: pose };
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

    owner ??= closestMappedPlayerId(wrist, faceResult.faceLandmarks, facePlayerIds, 0.32);
    if (owner !== undefined) addCue(owner, cue);
  }

  return cuesByPlayer;
}

// Require two detected frames to activate and two missing frames to end a cue.
function scoreCueEvents(cuesByPlayer, now) {
  let changed = false;
  for (const track of poseTracks.values()) {
    const detectedCues = cuesByPlayer.get(track.id) ?? new Set();
    const playerScore = ensurePlayerScore(track.id);
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
          playerScore.actions[cueId] = (playerScore.actions[cueId] ?? 0) + 1;
          changed = true;

          const lastAwardAt = track.lastAwardAt.get(cueId) ?? -Infinity;
          if (now - lastAwardAt >= CUE_COOLDOWN_MS) {
            playerScore.points += getPoints(cueId);
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
    const playerTracks = matchPoses(poseResult.landmarks, now);
    const facePlayerIds = matchFacesToPlayers(
      faceResult.faceLandmarks,
      poseResult.landmarks,
      playerTracks,
    );
    const cuesByPlayer = collectCueEvents(
      faceResult,
      handResult,
      poseResult,
      facePlayerIds,
      playerTracks,
    );

    faceCount.value = String(faceResult.faceLandmarks.length);
    handCount.value = String(handResult.landmarks.length);
    poseCount.value = String(poseResult.landmarks.length);
    scoreCueEvents(cuesByPlayer, now);

    drawFaceMesh(faceResult, video, overlay, overlayContext, drawingUtils, facePlayerIds);
    drawPoses(poseResult, drawingUtils);
    drawHands(handResult, drawingUtils);
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

    stopButton.disabled = false;
    lastInferenceAt = 0;
    setStatus("Camera live. Points and player numbers are saved on this device only.");
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
  poseTracks.clear();
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
    playerScores.clear();
    unassignedScores = [];
    nextPlayerId = 1;
    poseTracks.clear();
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

// Connect page actions and initialize the scoreboard without rewriting cleared settings.
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", () => stopCamera());
deleteSavedDataButton.addEventListener("click", deleteSavedData);
window.addEventListener("pagehide", () => stopCamera("Camera closed."));

updatePointControls(false);
updateScoreboard();
