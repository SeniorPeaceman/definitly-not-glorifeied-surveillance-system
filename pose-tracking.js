import { PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

export function createPoseLandmarker(vision, maxPoses) {
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL },
    runningMode: "VIDEO",
    numPoses: maxPoses,
  });
}

export function classifyPose(landmarks) {
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftWrist = landmarks[15];
  const rightWrist = landmarks[16];
  const visible = (landmark) => landmark && (landmark.visibility ?? 1) >= 0.5;

  if (
    visible(leftShoulder) && visible(rightShoulder) &&
    visible(leftWrist) && visible(rightWrist) &&
    leftWrist.y < leftShoulder.y - 0.05 &&
    rightWrist.y < rightShoulder.y - 0.05
  ) {
    return [{ key: "both-arms-raised", label: "Both arms raised" }];
  }

  return [];
}

export function drawPose(result, drawingUtils) {
  for (const landmarks of result.landmarks) {
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      color: "#ffb000",
      lineWidth: 3,
    });
  }
}