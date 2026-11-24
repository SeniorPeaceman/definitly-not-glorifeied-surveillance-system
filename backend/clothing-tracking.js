const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3/dist/transformers.min.js";
const MODEL_ID = "Xenova/segformer_b2_clothes";
const CLOTHING_INTERVAL_MS = 6000;

let segmenter;

export async function createClothingSegmenter(onProgress) {
  if (segmenter) return segmenter;
  const { pipeline } = await import(TRANSFORMERS_URL);
  segmenter = await pipeline("image-segmentation", MODEL_ID, {
    dtype: "q8",
    progress_callback: onProgress,
  });
  return segmenter;
}

export function getClothingIntervalMs() {
  return CLOTHING_INTERVAL_MS;
}

function isYellow(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const difference = max - min;
  let hue = 0;

  if (difference !== 0) {
    if (max === r) hue = 60 * (((g - b) / difference) % 6);
    else if (max === g) hue = 60 * ((b - r) / difference + 2);
    else hue = 60 * ((r - g) / difference + 4);
  }
  if (hue < 0) hue += 360;

  const saturation = max === 0 ? 0 : difference / max;
  const value = max / 255;
  return hue >= 38 && hue <= 72 && saturation >= 0.35 && value >= 0.25;
}

function isBlue(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const difference = max - min;
  let hue = 0;

  if (difference !== 0) {
    if (max === r) hue = 60 * (((g - b) / difference) % 6);
    else if (max === g) hue = 60 * ((b - r) / difference + 2);
    else hue = 60 * ((r - g) / difference + 4);
  }
  if (hue < 0) hue += 360;

  const saturation = max === 0 ? 0 : difference / max;
  const value = max / 255;
  return hue >= 185 && hue <= 245 && saturation >= 0.35 && value >= 0.25;
}

export async function classifyClothing(video, poseTracks, canvas, context) {
  if (!segmenter) throw new Error("Clothing segmenter is not loaded.");

  const canvasWidth = Math.min(320, video.videoWidth || 320);
  const canvasHeight = Math.round(canvasWidth * ((video.videoHeight || 240) / (video.videoWidth || 320)));
  if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const segments = await segmenter(canvas, {
    subtask: "semantic",
    target_sizes: [[canvas.height, canvas.width]],
  });

  const upperSegment = segments.find(({ label }) => label === "Upper-clothes");
  if (!upperSegment?.mask) return new Map();

  const mask = upperSegment.mask;
  if (!mask.data || mask.width !== canvas.width || mask.height !== canvas.height) {
    await mask.resize(canvas.width, canvas.height);
  }

  const imageData = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const maskData = mask.data;
  const maskChannels = mask.channels;
  const cuesByPlayer = new Map([...poseTracks.values()].map((track) => [track.id, new Set()]));

  for (const track of poseTracks.values()) {
    const left = Math.max(0, Math.floor((track.centerX - Math.max(track.width, 0.12)) * canvas.width));
    const right = Math.min(canvas.width, Math.ceil((track.centerX + Math.max(track.width, 0.12)) * canvas.width));
    const top = Math.max(0, Math.floor((track.centerY - track.height) * canvas.height));
    const bottom = Math.min(canvas.height, Math.ceil((track.centerY + track.height * 0.25) * canvas.height));

    let yellowCount = 0;
    let blueCount = 0;
    let upperCount = 0;

    for (let y = top; y < bottom; y += 2) {
      for (let x = left; x < right; x += 2) {
        const pixelIndex = y * canvas.width + x;
        const maskIndex = maskChannels === 1
          ? pixelIndex
          : pixelIndex * maskChannels;
        if ((maskData[maskIndex] ?? 0) < 128) continue;

        upperCount += 1;
        const imageIndex = pixelIndex * 4;
        const r = imageData[imageIndex];
        const g = imageData[imageIndex + 1];
        const b = imageData[imageIndex + 2];
        if (isYellow(r, g, b)) yellowCount += 1;
        if (isBlue(r, g, b)) blueCount += 1;
      }
    }

    if (upperCount < 20) continue;
    if (yellowCount / upperCount >= 0.12) cuesByPlayer.get(track.id).add("yellow-jacket");
    if (blueCount / upperCount >= 0.12) cuesByPlayer.get(track.id).add("blue-tshirt");
  }

  return cuesByPlayer;
}
