import { FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// This module creates the face model, classifies facial cues, and draws face tracks.
// Model file used for face landmarks and expression cues.
const FACE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Create the face landmarker that supplies facial points and expression cues.
export function createFaceLandmarker(vision, maxFaces) {
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL },
    runningMode: "VIDEO",
    numFaces: maxFaces,
    outputFaceBlendshapes: true,
  });
}

// Look up a named expression score, returning zero if it is unavailable.
function getBlendshapeScore(blendshapes, name) {
  return blendshapes.categories.find((category) => category.categoryName === name)?.score ?? 0;
}

// Turn selected face blendshapes into weighted cues for the live score display.
export function classifyFace(blendshapes) {
  const jawOpen = getBlendshapeScore(blendshapes, "jawOpen");
  const browDown = Math.max(
    getBlendshapeScore(blendshapes, "browDownLeft"),
    getBlendshapeScore(blendshapes, "browDownRight"),
  );
  const cues = [];

  if (jawOpen >= 0.5) cues.push("open-mouth");
  if (browDown >= 0.5) cues.push("lowered-brows");
  return cues;
}

// Return a normalized bounding box and center for matching faces to pose tracks.
export function getFaceBounds(landmarks) {
  const xs = landmarks.map(({ x }) => x);
  const ys = landmarks.map(({ y }) => y);
  const left = Math.min(...xs);
  const top = Math.min(...ys);
  const right = Math.max(...xs);
  const bottom = Math.max(...ys);

  return {
    left,
    top,
    right,
    bottom,
    centerX: (left + right) / 2,
    centerY: (top + bottom) / 2,
    width: right - left,
    height: bottom - top,
  };
}

// Draw only the face mesh onto the mirrored video overlay.
export function drawFaceMesh(result, video, canvas, context, drawingUtils) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);

  for (const landmarks of result.faceLandmarks) {
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#ff6b4a", lineWidth: 1 },
    );
  }
}

// Draw one pose-owned player frame per track, including frames for covered faces.
export function drawPlayerFrames(playerFrames, video, canvas, context) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  for (const frame of playerFrames) {
    if (!frame?.bounds) continue;
    const left = frame.bounds.left * canvas.width;
    const top = frame.bounds.top * canvas.height;
    const width = (frame.bounds.right - frame.bounds.left) * canvas.width;
    const height = (frame.bounds.bottom - frame.bounds.top) * canvas.height;

    context.save();
    context.strokeStyle = "#ff6b4a";
    context.lineWidth = Math.max(2, canvas.width / 320);
    context.setLineDash([]);
    context.strokeRect(left, top, width, height);

    const timerSeconds = frame.remainingMs / 1000;
    const timerText = `${String(Math.floor(Math.max(0, timerSeconds) / 60)).padStart(2, "0")}:${String(Math.max(0, Math.ceil(timerSeconds) % 60)).padStart(2, "0")}`;
    const label = `PLAYER ${frame.playerId} | ${timerText}`;
    const fontSize = Math.max(14, canvas.width / 34);
    const labelTop = Math.max(0, top - fontSize - 10);
    context.setLineDash([]);
    context.font = `700 ${fontSize}px monospace`;
    context.textBaseline = "middle";
    const labelWidth = context.measureText(label).width + 16;
    context.fillStyle = "#ff6b4a";
    context.fillRect(left, labelTop, labelWidth, fontSize + 8);
    context.fillStyle = "#17251e";
    const textLeft = left + 8;
    const textWidth = context.measureText(label).width;
    context.translate(textLeft * 2 + textWidth, 0);
    context.scale(-1, 1);
    context.fillText(label, textLeft, labelTop + (fontSize + 8) / 2);
    context.restore();
  }
}
