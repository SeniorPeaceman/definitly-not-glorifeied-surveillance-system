import { PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

export function createPoseLandmarker(vision, maxPoses) {
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL },
    runningMode: "VIDEO",
    numPoses: maxPoses,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

function isVisible(landmark) {
  return landmark && (landmark.visibility ?? 1) >= 0.45;
}

export function classifyPose(landmarks) {
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftWrist = landmarks[15];
  const rightWrist = landmarks[16];
  const cues = [];

  if (
    isVisible(leftShoulder) &&
    isVisible(rightShoulder) &&
    isVisible(leftWrist) &&
    isVisible(rightWrist)
  ) {
    if (leftWrist.y < leftShoulder.y - 0.06 && rightWrist.y < rightShoulder.y - 0.06) {
      cues.push("both-arms-raised");
    }

    const shoulderWidth = Math.abs(rightShoulder.x - leftShoulder.x);
    const wristWidth = Math.abs(rightWrist.x - leftWrist.x);
    if (shoulderWidth > 0 && wristWidth > shoulderWidth * 1.65) {
      cues.push("arms-spread");
    }
  }

  return cues;
}

export function drawPoses(result, drawingUtils) {
  for (const landmarks of result.landmarks) {
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      color: "#58e0bd",
      lineWidth: 3,
    });
    drawingUtils.drawLandmarks(landmarks, {
      color: "#ffca58",
      lineWidth: 1,
      radius: 3,
    });
  }
}