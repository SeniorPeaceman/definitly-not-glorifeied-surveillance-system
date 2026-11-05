import { FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

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

  if (jawOpen >= 0.5) cues.push({ weight: "#weight-open-mouth", label: "Wide open mouth" });
  if (browDown >= 0.5) cues.push({ weight: "#weight-lowered-brows", label: "Lowered brows (frown proxy)" });
  return cues;
}

// Draw facial tessellation and a padded bounding box on the tracking canvas.
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
      { color: "#00ff00", lineWidth: 0.5 },
    );

    const xs = landmarks.map(({ x }) => x * canvas.width);
    const ys = landmarks.map(({ y }) => y * canvas.height);
    const left = Math.min(...xs);
    const top = Math.min(...ys);
    const right = Math.max(...xs);
    const bottom = Math.max(...ys);
    const padding = Math.max(right - left, bottom - top) * 0.05;

    context.strokeStyle = "#ff0000";
    context.lineWidth = Math.max(2, canvas.width / 320);
    context.strokeRect(
      left - padding,
      top - padding,
      right - left + padding * 2,
      bottom - top + padding * 2,
    );
  }
}
