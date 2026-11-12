import { PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// This module detects visible body poses and renders their skeleton connections.
const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

// Create the video-mode body-pose detector with the configured person limit.
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

// Ignore landmarks that the model considers too uncertain to use for scoring.
function isVisible(landmark) {
  return landmark && (landmark.visibility ?? 1) >= 0.45;
}

// Convert raised arms and wide arm spacing into stable game cue IDs.
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

// Draw the detected body skeletons over the live camera frame.
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

export function drawPlayerOutlines(result, canvas, context, tracksByPose) {
  for (const [index, landmarks] of result.landmarks.entries()) {
    const bodyPoints = landmarks.slice(11).filter(isVisible);
    if (bodyPoints.length < 4) continue;

    const track = tracksByPose[index];
    if (!track) continue;
    const paddingX = Math.max(...bodyPoints.map(({ x }) => x)) - Math.min(...bodyPoints.map(({ x }) => x));
    const paddingY = Math.max(...bodyPoints.map(({ y }) => y)) - Math.min(...bodyPoints.map(({ y }) => y));
    const left = (Math.min(...bodyPoints.map(({ x }) => x)) - paddingX * 0.04) * canvas.width;
    const top = (Math.min(...bodyPoints.map(({ y }) => y)) - paddingY * 0.04) * canvas.height;
    const width = (paddingX * 1.08) * canvas.width;
    const height = (paddingY * 1.08) * canvas.height;
    const perimeter = 2 * (width + height);
    const progress = Math.max(0, Math.min(1, track.progress));

    context.save();
    context.lineWidth = Math.max(3, canvas.width / 240);
    context.strokeStyle = track.remainingMs === 0 ? "rgba(255, 48, 48, 0.3)" : "rgba(255, 255, 255, 0.25)";
    context.setLineDash([]);
    context.strokeRect(left, top, width, height);
    context.setLineDash([Math.max(0.001, perimeter * progress), perimeter]);
    context.strokeStyle = track.remainingMs === 0 ? "#ff3030" : "#ffe35a";
    context.strokeRect(left, top, width, height);
    context.restore();
  }
}
