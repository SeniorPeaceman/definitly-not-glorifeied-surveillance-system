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
  };
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

    const bounds = getFaceBounds(landmarks);
    const left = bounds.left * canvas.width;
    const top = bounds.top * canvas.height;
    const width = (bounds.right - bounds.left) * canvas.width;
    const height = (bounds.bottom - bounds.top) * canvas.height;
    const padding = Math.max(width, height) * 0.05;

    context.strokeStyle = "#ff0000";
    context.lineWidth = Math.max(2, canvas.width / 320);
    context.strokeRect(
      left - padding,
      top - padding,
      width + padding * 2,
      height + padding * 2,
    );
  }
}
