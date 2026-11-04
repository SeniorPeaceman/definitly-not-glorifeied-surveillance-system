import { HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// Model file used to find hands and produce their landmark points.
const HAND_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Create a video-mode hand tracker limited to the requested number of hands.
export function createHandLandmarker(vision, maxHands) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL_URL },
    runningMode: "VIDEO",
    numHands: maxHands,
  });
}

// Measure finger bend at a joint; angles stay comparable as the hand turns.
function jointAngle(first, joint, last) {
  const firstVector = {
    x: first.x - joint.x,
    y: first.y - joint.y,
    z: first.z - joint.z,
  };
  const lastVector = {
    x: last.x - joint.x,
    y: last.y - joint.y,
    z: last.z - joint.z,
  };
  const firstLength = Math.hypot(firstVector.x, firstVector.y, firstVector.z);
  const lastLength = Math.hypot(lastVector.x, lastVector.y, lastVector.z);
  if (firstLength === 0 || lastLength === 0) return undefined;

  const dotProduct =
    firstVector.x * lastVector.x +
    firstVector.y * lastVector.y +
    firstVector.z * lastVector.z;
  const cosine = Math.max(-1, Math.min(1, dotProduct / (firstLength * lastLength)));
  return Math.acos(cosine) * (180 / Math.PI);
}

// Approximate selected gestures by checking finger extension and hand position.
export function classifyGesture(landmarks) {
  const fingers = [
    { mcp: 5, pip: 6, dip: 7 },
    { mcp: 9, pip: 10, dip: 11 },
    { mcp: 13, pip: 14, dip: 15 },
    { mcp: 17, pip: 18, dip: 19 },
  ];
  const bendAngles = fingers.map(({ mcp, pip, dip }) =>
    jointAngle(landmarks[mcp], landmarks[pip], landmarks[dip]),
  );
  // Leave intermediate poses unclassified to avoid flickering around one cutoff.
  const curled = bendAngles.map((angle) => angle !== undefined && angle < 145);
  const extended = bendAngles.map((angle) => angle !== undefined && angle > 160);
  const middleFingerRaised = extended[1] && curled[0] && curled[2] && curled[3];

  if (middleFingerRaised) return { weight: "#weight-middle-finger", label: "Raised middle finger" };
  if (curled.every(Boolean)) {
    // Hand height is used as a simple proxy to distinguish a raised fist.
    const upperFrameFist = landmarks[0].y < 0.45;
    return upperFrameFist
      ? { weight: "#weight-raised-fist", label: "Raised fist (upper-frame proxy)" }
      : { weight: "#weight-fist", label: "Fist" };
  }
  return undefined;
}

// Draw hand connections and landmark points on the tracking canvas.
export function drawHands(result, drawingUtils) {
  for (const landmarks of result.landmarks) {
    drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "#00bfff",
      lineWidth: 3,
    });
    drawingUtils.drawLandmarks(landmarks, {
      color: "#ff00ff",
      lineWidth: 1,
      radius: 3,
    });
  }
}
