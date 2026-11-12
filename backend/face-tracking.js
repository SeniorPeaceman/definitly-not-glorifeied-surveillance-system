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

// Return a normalized bounding box and center used to keep a face's track number stable.
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

// Draw the face mesh, track box, and readable ID label onto the mirrored video overlay.
export function drawFaceMesh(result, video, canvas, context, drawingUtils, faceIds = []) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);

  for (const [index, landmarks] of result.faceLandmarks.entries()) {
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#00ff00", lineWidth: 0.5 },
    );

    const bounds = getFaceBounds(landmarks);
    const left = bounds.left * canvas.width;
    const top = bounds.top * canvas.height;
    const right = bounds.right * canvas.width;
    const bottom = bounds.bottom * canvas.height;
    const padding = Math.max(right - left, bottom - top) * 0.05;

    context.strokeStyle = "#ff6b4a";
    context.lineWidth = Math.max(2, canvas.width / 320);
    context.strokeRect(
      left - padding,
      top - padding,
      right - left + padding * 2,
      bottom - top + padding * 2,
    );

    const label = `FACE ${faceIds[index] ?? index + 1}`;
    const fontSize = Math.max(14, canvas.width / 34);
    const labelTop = Math.max(0, top - padding - fontSize - 10);
    context.font = `700 ${fontSize}px monospace`;
    context.textBaseline = "middle";
    const labelWidth = context.measureText(label).width + 16;
    context.fillStyle = "#ff6b4a";
    context.fillRect(left - padding, labelTop, labelWidth, fontSize + 8);
    context.fillStyle = "#17251e";
    const textLeft = left - padding + 8;
    const textWidth = context.measureText(label).width;
    context.save();
    context.translate(textLeft * 2 + textWidth, 0);
    context.scale(-1, 1);
    context.fillText(label, textLeft, labelTop + (fontSize + 8) / 2);
    context.restore();
  }
}
