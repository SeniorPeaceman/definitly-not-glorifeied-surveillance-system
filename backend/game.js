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
const PLAYER_TIMER_MS = 120000;
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
      return {
        id: score.id,
        points: score.points,
        actions,
        remainingMs: Number.isFinite(score.remainingMs)
          ? Math.max(0, Math.min(PLAYER_TIMER_MS, score.remainingMs))
          : PLAYER_TIMER_MS,
        elapsedMs: Number.isFinite(score.elapsedMs) ? Math.max(0, score.elapsedMs) : 0,
        lost: Boolean(score.lost),
      };
    });
    const unassignedScores = Array.isArray(saved?.unassignedScores)
      ? saved.unassignedScores.filter(Number.isFinite)
      : [];
    const highestId = playerScores.reduce((highest, score) => Math.max(highest, score.id), 0);
    return {
      nextPlayerId: Math.max(highestId + 1, Number(saved?.nextPlayerId ?? saved?.nextFaceId) || 1),
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
// Keep active camera and pose-track state in memory; player records are persisted.
let nextPlayerId = savedGame.nextPlayerId;
let storageEnabled = true;
let faceLandmarker;
let handLandmarker;
let poseLandmarker;
let cameraStream;
let animationFrameId;
let lastInferenceAt = 0;
const playerTracks = new Map();
const lostPlayerIds = new Set();
let lastTimerSaveAt = 0;
const colorSampleCanvas = document.createElement("canvas");
colorSampleCanvas.width = 24;
colorSampleCanvas.height = 24;
const colorSampleContext = colorSampleCanvas.getContext("2d", { willReadFrequently: true });

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
const criminalAlert = document.querySelector("#criminal-alert");
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

// Create a fresh player record when pose tracking first assigns a number.
function ensurePlayerScore(id) {
  if (!playerScores.has(id)) {
    playerScores.set(id, {
      id,
      points: 0,
      actions: [],
      remainingMs: PLAYER_TIMER_MS,
      elapsedMs: 0,
      lost: false,
    });
  }
  return playerScores.get(id);
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000);
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

// Render each face's number, point total, and count of each detected action.
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
    empty.textContent = "No players yet. Start the camera to begin.";
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
    const remaining = `${formatDuration(scoreEntry.remainingMs)} remaining`;
    const elapsed = `${formatDuration(scoreEntry.elapsedMs)} played`;
    const status = scoreEntry.lost ? "LOST" : remaining;
    item.textContent = scoreEntry.kind === "player"
      ? `Player (${scoreEntry.id}), ${status}, ${elapsed}, Points: ${scoreEntry.points}, List of actions: ${actions}`
      : `Unassigned total (${scoreEntry.index + 1}), Points: ${scoreEntry.points}, List of actions: ${actions}`;
    scoreList.append(item);
  }
}

// Mirror slider values in their outputs and optionally persist the settings.
function updatePointControls(save = true) {
  for (const cue of CUES) {
    const input = document.querySelector(`#${cue.input}`);
    const output = document.querySelector(`#${cue.input}-value`);
    output.value = `-${input.value}s`;
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

function poseAnchor(landmarks) {
  const torso = [11, 12, 23, 24]
    .map((index) => landmarks[index])
    .filter((point) => point && (point.visibility ?? 1) >= 0.4);
  if (torso.length < 2) return undefined;
  return {
    x: torso.reduce((total, point) => total + point.x, 0) / torso.length,
    y: torso.reduce((total, point) => total + point.y, 0) / torso.length,
  };
}

// Number players from pose detections, keeping IDs through short tracking gaps.
function matchPoses(poses, now) {
  for (const [id, track] of playerTracks) {
    if (now - track.lastSeen >= TRACK_TIMEOUT_MS) playerTracks.delete(id);
  }
  const detections = poses.map(poseAnchor);
  const tracks = [...playerTracks.values()];
  const costMatrix = detections.map((anchor) => {
    const trackCosts = tracks.map((track) => {
      if (!anchor) return 100;
      const elapsed = Math.min((now - track.lastSeen) / 1000, 0.75);
      const predictedX = track.centerX + track.velocityX * elapsed;
      const predictedY = track.centerY + track.velocityY * elapsed;
      const distance = Math.hypot(anchor.x - predictedX, anchor.y - predictedY);
      const distanceLimit = 0.2 + Math.min(0.12, Math.hypot(track.velocityX, track.velocityY) * elapsed * 0.4);
      return distance > distanceLimit ? 100 : distance / distanceLimit;
    });
    return [...trackCosts, ...new Array(detections.length).fill(1.1)];
  });
  const assignments = assignMinimumCostPairs(costMatrix);

  return detections.map((anchor, detectionIndex) => {
    if (!anchor) return undefined;
    const trackIndex = assignments[detectionIndex];
    const track = trackIndex >= 0 && trackIndex < tracks.length ? tracks[trackIndex] : undefined;
    if (track) {
      const elapsed = Math.max((now - track.lastSeen) / 1000, INFERENCE_INTERVAL_MS / 1000);
      const measuredVelocityX = (anchor.x - track.centerX) / elapsed;
      const measuredVelocityY = (anchor.y - track.centerY) / elapsed;
      track.velocityX = track.velocityX * 0.6 + Math.max(-1.5, Math.min(1.5, measuredVelocityX)) * 0.4;
      track.velocityY = track.velocityY * 0.6 + Math.max(-1.5, Math.min(1.5, measuredVelocityY)) * 0.4;
      track.centerX = anchor.x;
      track.centerY = anchor.y;
      track.lastSeen = now;
      return track;
    }

    const id = nextPlayerId++;
    const score = ensurePlayerScore(id);
    const newTrack = {
      id,
      centerX: anchor.x,
      centerY: anchor.y,
      velocityX: 0,
      velocityY: 0,
      lastSeen: now,
      lastTimerUpdate: now,
      remainingMs: score.remainingMs,
      elapsedMs: score.elapsedMs,
      lost: score.lost,
      progress: 1 - score.remainingMs / PLAYER_TIMER_MS,
      camouflaged: false,
      cueStates: new Map(),
      lastAwardAt: new Map(),
    };
    playerTracks.set(id, newTrack);
    saveGame();
    return newTrack;
  });
}

// Associate a face or hand point with the nearest visible pose track.
function closestPlayerId(point, poses, tracks, maxDistance) {
  if (!point) return undefined;
  let closestId;
  let closestDistance = maxDistance;

  for (let index = 0; index < poses.length; index += 1) {
    const track = tracks[index];
    if (!track) continue;
    const landmarks = poses[index];
    const candidates = [0, 11, 12, 15, 16, 23, 24]
      .map((landmarkIndex) => landmarks[landmarkIndex])
      .filter((landmark) => landmark && (landmark.visibility ?? 1) >= 0.35);
    for (const candidate of candidates) {
      const distance = Math.hypot(point.x - candidate.x, point.y - candidate.y);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestId = track.id;
      }
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

function faceCenter(landmarks) {
  const bounds = getFaceBounds(landmarks);
  return { x: bounds.centerX, y: bounds.centerY };
}

// Combine facial expressions, body poses, and hand gestures under a pose track.
function collectCueEvents(faceResult, handResult, poseResult, tracks) {
  const cuesByPlayer = new Map(tracks.filter(Boolean).map((track) => [track.id, new Set()]));
  const addCue = (id, cue) => cuesByPlayer.get(id)?.add(cue);

  for (const [index, blendshapes] of faceResult.faceBlendshapes.entries()) {
    if (blendshapes) {
      const id = closestPlayerId(
        faceCenter(faceResult.faceLandmarks[index]),
        poseResult.landmarks,
        tracks,
        0.3,
      );
      if (id !== undefined) {
        for (const cue of classifyFace(blendshapes)) addCue(id, cue);
      }
    }
  }

  const poseOwners = poseResult.landmarks.map((pose, index) => {
    const id = tracks[index]?.id;
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

    owner ??= closestPlayerId(wrist, poseResult.landmarks, tracks, 0.4);
    if (owner !== undefined) addCue(owner, cue);
  }

  return cuesByPlayer;
}

function detectColorCamouflage(pose) {
  if (!colorSampleContext || video.videoWidth === 0) return undefined;
  const leftShoulder = pose[11];
  const rightShoulder = pose[12];
  const leftHip = pose[23];
  const rightHip = pose[24];
  if (![leftShoulder, rightShoulder, leftHip, rightHip].every((point) => point && (point.visibility ?? 1) >= 0.4)) {
    return undefined;
  }

  const centerX = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4;
  const sourceX = Math.max(0, Math.min(video.videoWidth - 1, (centerX - Math.abs(rightShoulder.x - leftShoulder.x) * 0.275) * video.videoWidth));
  const sourceY = Math.max(0, Math.min(video.videoHeight - 1, Math.min(leftShoulder.y, rightShoulder.y) * video.videoHeight));
  const sourceWidth = Math.max(1, Math.min(video.videoWidth - sourceX, Math.abs(rightShoulder.x - leftShoulder.x) * 0.55 * video.videoWidth));
  const torsoBottom = ((leftHip.y + rightHip.y) / 2) * video.videoHeight;
  if (torsoBottom <= sourceY) return undefined;
  const sourceHeight = Math.max(1, Math.min(video.videoHeight - sourceY, torsoBottom - sourceY));
  colorSampleContext.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, colorSampleCanvas.width, colorSampleCanvas.height);

  const pixels = colorSampleContext.getImageData(0, 0, colorSampleCanvas.width, colorSampleCanvas.height).data;
  let yellowPixels = 0;
  let bluePixels = 0;
  let saturatedPixels = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    const red = pixels[index] / 255;
    const green = pixels[index + 1] / 255;
    const blue = pixels[index + 2] / 255;
    const maximum = Math.max(red, green, blue);
    const minimum = Math.min(red, green, blue);
    if (maximum < 0.18 || maximum === minimum || (maximum - minimum) / maximum < 0.38) continue;
    saturatedPixels += 1;
    const hue = maximum === red
      ? 60 * (((green - blue) / (maximum - minimum) + 6) % 6)
      : maximum === green
        ? 60 * ((blue - red) / (maximum - minimum) + 2)
        : 60 * ((red - green) / (maximum - minimum) + 4);
    if (hue >= 42 && hue <= 72) yellowPixels += 1;
    if (hue >= 190 && hue <= 250) bluePixels += 1;
  }
  if (saturatedPixels === 0) return undefined;
  if (yellowPixels / saturatedPixels >= 0.28) return "yellow";
  if (bluePixels / saturatedPixels >= 0.28) return "blue";
  return undefined;
}

function updateCriminalAlert() {
  criminalAlert.hidden = lostPlayerIds.size === 0;
  criminalAlert.textContent = lostPlayerIds.size
    ? `CRIMINAL RECOGNIZED · PLAYER ${[...lostPlayerIds].sort((first, second) => first - second).join(", ")}`
    : "";
}

function updatePlayerTimers(tracks, poses, now) {
  let changed = false;
  for (const [index, track] of tracks.entries()) {
    if (!track) continue;
    const elapsed = Math.max(0, now - track.lastTimerUpdate);
    track.lastTimerUpdate = now;
    track.camouflaged = Boolean(detectColorCamouflage(poses[index]));
    track.elapsedMs += elapsed;
    if (!track.camouflaged && !track.lost) track.remainingMs = Math.max(0, track.remainingMs - elapsed);
    track.progress = 1 - track.remainingMs / PLAYER_TIMER_MS;

    const score = ensurePlayerScore(track.id);
    score.remainingMs = track.remainingMs;
    score.elapsedMs = track.elapsedMs;
    if (track.remainingMs === 0 && !track.lost) {
      track.lost = true;
      score.lost = true;
      lostPlayerIds.add(track.id);
      changed = true;
    }
    if (track.lost) score.lost = true;
    changed = true;
  }

  updateCriminalAlert();
  if (changed && now - lastTimerSaveAt >= 1000) {
    lastTimerSaveAt = now;
    saveGame();
  }
}

// Require two detected frames to activate and two missing frames to end a cue.
function scoreCueEvents(cuesByPlayer, now) {
  let changed = false;
  for (const track of playerTracks.values()) {
    if (track.lost) continue;
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
            const penaltyMs = getPoints(cueId) * 1000;
            playerScore.points += getPoints(cueId);
            track.remainingMs = Math.max(0, track.remainingMs - penaltyMs);
            playerScore.remainingMs = track.remainingMs;
            track.progress = 1 - track.remainingMs / PLAYER_TIMER_MS;
            track.lastAwardAt.set(cueId, now);
            changed = true;
            if (track.remainingMs === 0 && !track.lost) {
              track.lost = true;
              playerScore.lost = true;
              lostPlayerIds.add(track.id);
            }
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

  if (changed) {
    updateCriminalAlert();
    saveGame();
  }
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
    const tracks = matchPoses(poseResult.landmarks, now);
    const cuesByPlayer = collectCueEvents(faceResult, handResult, poseResult, tracks);

    faceCount.value = String(faceResult.faceLandmarks.length);
    handCount.value = String(handResult.landmarks.length);
    poseCount.value = String(poseResult.landmarks.length);
    updatePlayerTimers(tracks, poseResult.landmarks, now);
    scoreCueEvents(cuesByPlayer, now);

    const faceIds = faceResult.faceLandmarks.map((landmarks) => {
      return closestPlayerId(
        faceCenter(landmarks),
        poseResult.landmarks,
        tracks,
        0.3,
      );
    });
    drawFaceMesh(faceResult, video, overlay, overlayContext, drawingUtils, faceIds);
    drawPoses(poseResult, drawingUtils);
    drawHands(handResult, drawingUtils);
    drawPlayerOutlines(poseResult, overlay, overlayContext, tracks);
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
    setStatus("Camera live. Player timers and scores are saved on this device only.");
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
  saveGame();
  cameraStream = undefined;
  video.srcObject = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  playerTracks.clear();
  lostPlayerIds.clear();
  updateCriminalAlert();
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
    playerTracks.clear();
    lostPlayerIds.clear();
    updateCriminalAlert();
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
