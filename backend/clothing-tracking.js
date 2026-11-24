import { ImageClassifier } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const LONG_SLEEVE_LABELS = [
  "long sleeve top",
  "long sleeve outwear",
  "top",
  "shirt",
  "sweater",
  "pullover",
];

export async function createClothingClassifier(vision, modelAssetBuffer) {
  return ImageClassifier.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer },
    runningMode: "IMAGE",
    scoreThreshold: 0.45,
  });
}

function rgbToBlueRatio(red, green, blue) {
  const maximum = Math.max(red, green, blue);
  const minimum = Math.min(red, green, blue);
  if (maximum - minimum < 35) return 0;

  let hue = 0;
  if (maximum === blue) {
    hue = 60 * ((red - green) / (maximum - minimum) + 4);
  } else {
    return 0;
  }
  if (hue < 0) hue += 360;

  const blueDominant = blue > red * 1.25 && blue > green * 1.08;
  return blueDominant && hue >= 190 && hue <= 250 ? 1 : 0;
}

function isLongSleeveCategory(categoryName) {
  const normalized = categoryName.trim().toLowerCase();
  return LONG_SLEEVE_LABELS.some((label) => normalized.includes(label));
}

function torsoBounds(pose, video) {
  const torso = [11, 12, 23, 24].map((index) => pose[index])
    .filter((point) => point && (point.visibility ?? 1) >= 0.35);
  if (torso.length < 2) return undefined;

  const xs = torso.map(({ x }) => x);
  const ys = torso.map(({ y }) => y);
  const left = Math.max(0, Math.min(...xs) - 0.12);
  const right = Math.min(1, Math.max(...xs) + 0.12);
  const top = Math.max(0, Math.min(...ys) - 0.12);
  const bottom = Math.min(1, Math.max(...ys) + 0.12);
  return {
    left: left * video.videoWidth,
    top: top * video.videoHeight,
    width: (right - left) * video.videoWidth,
    height: (bottom - top) * video.videoHeight,
  };
}

export function classifyPlayerClothing(classifier, poseLandmarks, playerTracks, video) {
  if (!classifier || !video.videoWidth || !video.videoHeight) return [];
  const cropCanvas = document.createElement("canvas");
  const cropContext = cropCanvas.getContext("2d", { willReadFrequently: true });
  const results = [];

  for (const [index, pose] of poseLandmarks.entries()) {
    const track = playerTracks[index];
    const bounds = torsoBounds(pose, video);
    if (!track || !bounds || bounds.width < 12 || bounds.height < 12) continue;

    cropCanvas.width = Math.round(bounds.width);
    cropCanvas.height = Math.round(bounds.height);
    cropContext.drawImage(
      video,
      bounds.left,
      bounds.top,
      bounds.width,
      bounds.height,
      0,
      0,
      cropCanvas.width,
      cropCanvas.height,
    );
    const classification = classifier.classify(cropCanvas);
    const category = classification.classifications?.[0]?.categories?.[0];
    if (!category || category.score < 0.45 || !isLongSleeveCategory(category.categoryName ?? "")) {
      continue;
    }

    let bluePixels = 0;
    let sampledPixels = 0;
    for (let y = 0; y < cropCanvas.height; y += 5) {
      for (let x = 0; x < cropCanvas.width; x += 5) {
        const pixel = cropContext.getImageData(x, y, 1, 1).data;
        bluePixels += rgbToBlueRatio(pixel[0], pixel[1], pixel[2]);
        sampledPixels += 1;
      }
    }

    if (sampledPixels > 0 && bluePixels / sampledPixels >= 0.14) {
      results.push({ playerId: track.id, category, bounds });
    }
  }
  return results;
}
