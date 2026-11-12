const MAX_IMAGE_BYTES = 2 * 1024 * 1024;

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin");
    const corsHeaders = {
      "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN ?? "null",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Vary": "Origin",
    };

    if (!env.ALLOWED_ORIGIN || origin !== env.ALLOWED_ORIGIN) {
      return new Response("Origin not allowed", { status: 403 });
    }
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405, headers: corsHeaders });
    }
    if (!request.headers.get("Content-Type")?.startsWith("image/jpeg")) {
      return new Response("Expected image/jpeg", { status: 415, headers: corsHeaders });
    }
    if (!env.ROBOFLOW_API_KEY || !env.ROBOFLOW_MODEL_ID || !env.ROBOFLOW_MODEL_VERSION) {
      return new Response("Roboflow Worker configuration is incomplete", {
        status: 503,
        headers: corsHeaders,
      });
    }

    const image = await request.arrayBuffer();
    if (image.byteLength === 0 || image.byteLength > MAX_IMAGE_BYTES) {
      return new Response("Image size is invalid", { status: 413, headers: corsHeaders });
    }

    const inferenceUrl = new URL(
      `https://serverless.roboflow.com/${env.ROBOFLOW_MODEL_ID}/${env.ROBOFLOW_MODEL_VERSION}`,
    );
    inferenceUrl.searchParams.set("api_key", env.ROBOFLOW_API_KEY);
    let inferenceResponse;
    try {
      inferenceResponse = await fetch(inferenceUrl, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: image,
      });
    } catch {
      return new Response(JSON.stringify({ error: "Roboflow is unavailable" }), {
        status: 502,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    return new Response(inferenceResponse.body, {
      status: inferenceResponse.status,
      headers: {
        ...corsHeaders,
        "Cache-Control": "no-store",
        "Content-Type": inferenceResponse.headers.get("Content-Type") ?? "application/json",
      },
    });
  },
};