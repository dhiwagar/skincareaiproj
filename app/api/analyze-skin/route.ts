import { GoogleGenAI, createPartFromBase64 } from "@google/genai";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type AnalysisProvider = {
  name: string;
  run: () => Promise<unknown>;
};

class ProviderError extends Error {
  constructor(
    public provider: string,
    message: string,
    public code?: string,
  ) {
    super(message);
  }
}

function parseDataUrl(dataUrl: string) {
  const match = dataUrl.match(/^data:(?<mimeType>[^;]+);base64,(?<data>.+)$/);

  if (!match?.groups) {
    return null;
  }

  return {
    mimeType: match.groups.mimeType,
    data: match.groups.data,
  };
}

function parseAnalysisJson(rawContent: string) {
  let cleanContent = rawContent.trim();

  if (cleanContent.includes("```json")) {
    cleanContent = cleanContent.split("```json")[1].split("```")[0].trim();
  } else if (cleanContent.includes("```")) {
    cleanContent = cleanContent.split("```")[1].split("```")[0].trim();
  }

  return JSON.parse(cleanContent);
}

function getErrorMessage(error: unknown) {
  if (!(error instanceof Error)) {
    return String(error);
  }

  try {
    const parsed = JSON.parse(error.message);
    return parsed.error?.message || error.message;
  } catch {
    return error.message;
  }
}

async function runOpenAiCompatibleProvider(params: {
  provider: string;
  apiUrl: string;
  apiKey: string;
  model: string;
  image: string;
  prompt: string;
}) {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${params.apiKey}`,
    "Content-Type": "application/json",
  };

  if (params.apiUrl.includes("openrouter.ai")) {
    headers["HTTP-Referer"] = "https://skin.chat";
    headers["X-Title"] = "Skin.Chat";
  }

  const response = await fetch(params.apiUrl, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: params.model,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: {
                url: params.image,
              },
            },
            { type: "text", text: params.prompt },
          ],
        },
      ],
      response_format:
        params.provider === "OpenAI" || params.provider === "Groq"
          ? { type: "json_object" }
          : undefined,
    }),
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    const error = data.error || {};
    throw new ProviderError(
      params.provider,
      error.message || `${params.provider} request failed`,
      error.code || error.type,
    );
  }

  const rawContent = data.choices?.[0]?.message?.content;

  if (!rawContent) {
    throw new ProviderError(params.provider, "No response from AI model");
  }

  return parseAnalysisJson(rawContent);
}

async function runGeminiProvider(params: {
  apiKey: string;
  model: string;
  image: string;
  prompt: string;
}) {
  const imagePart = parseDataUrl(params.image);

  if (!imagePart) {
    throw new ProviderError("Gemini", "Invalid image format");
  }

  const ai = new GoogleGenAI({ apiKey: params.apiKey });
  const response = await ai.models.generateContent({
    model: params.model,
    contents: [
      createPartFromBase64(imagePart.data, imagePart.mimeType),
      { text: params.prompt },
    ],
    config: {
      responseMimeType: "application/json",
    },
  });

  return parseAnalysisJson(response.text ?? "");
}

export async function POST(request: Request) {
  const providerErrors: string[] = [];

  try {
    const { image, concern } = await request.json();

    if (!image) {
      return NextResponse.json({ error: "Image is required for analysis" }, { status: 400 });
    }

    const prompt = `
      As a professional dermatological assistant, analyze the skin condition in the provided image.
      The user is specifically concerned about: ${concern || "General skin health"}.

      Provide a detailed analysis including hydration, oiliness, acne, and dark spots (as percentages 0-100).
      Provide a specific morning (AM) and evening (PM) routine.
      Include 4-5 recommended products available in India (under ₹500 range).

      IMPORTANT: Your entire response must be a single valid JSON object. Do not include any text before or after the JSON.

      JSON Structure:
      {
        "hydration": number,
        "oiliness": number,
        "acne": number,
        "dark_spots": number,
        "summary": "string",
        "tags": ["string"],
        "routine": {
          "am": ["string"],
          "pm": ["string"]
        },
        "recommended_products": [
          { "name": "string", "type": "string", "brand": "string", "price_range": "string" }
        ]
      }
    `;

    const openAiApiKey = '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';
    const groqApiKey = process.env.GROQ_API_KEY || process.env.GROK_API_KEY;
    const geminiApiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    const openRouterApiKey = process.env.OPENROUTER_API_KEY;
    const groqModel =
      process.env.GROQ_MODEL === "llama-3.2-11b-vision-preview" ? undefined : process.env.GROQ_MODEL;

    const providers: AnalysisProvider[] = [];

    if (openAiApiKey) {
      providers.push({
        name: "OpenAI",
        run: () =>
          runOpenAiCompatibleProvider({
            provider: "OpenAI",
            apiUrl: "https://api.openai.com/v1/chat/completions",
            apiKey: openAiApiKey,
            model: process.env.OPENAI_MODEL || "gpt-4o-mini",
            image,
            prompt,
          }),
      });
    }

    if (groqApiKey) {
      providers.push({
        name: "Groq",
        run: () =>
          runOpenAiCompatibleProvider({
            provider: "Groq",
            apiUrl: "https://api.groq.com/openai/v1/chat/completions",
            apiKey: groqApiKey,
            model: groqModel || "meta-llama/llama-4-scout-17b-16e-instruct",
            image,
            prompt,
          }),
      });
    }

    if (geminiApiKey) {
      providers.push({
        name: "Gemini",
        run: () =>
          runGeminiProvider({
            apiKey: geminiApiKey,
            model: process.env.GEMINI_MODEL || "gemini-2.5-flash",
            image,
            prompt,
          }),
      });
    }

    if (openRouterApiKey) {
      providers.push({
        name: "OpenRouter",
        run: () =>
          runOpenAiCompatibleProvider({
            provider: "OpenRouter",
            apiUrl: "https://openrouter.ai/api/v1/chat/completions",
            apiKey: openRouterApiKey,
            model:
              process.env.OPENROUTER_MODEL ||
              process.env.NEXT_PUBLIC_OPENROUTER_MODEL ||
              "google/gemini-2.5-flash",
            image,
            prompt,
          }),
      });
    }

    if (!providers.length) {
      return NextResponse.json({ error: "AI API key not configured" }, { status: 500 });
    }

    for (const provider of providers) {
      try {
        const analysis = await provider.run();
        return NextResponse.json(analysis);
      } catch (error) {
        const message = getErrorMessage(error);
        providerErrors.push(`${provider.name}: ${message}`);
        console.error(`${provider.name} analysis error:`, error);
      }
    }

    const quotaError = providerErrors.find((error) =>
      /insufficient_quota|quota|billing/i.test(error),
    );

    return NextResponse.json(
      {
        error:
          quotaError ||
          "All configured AI providers failed. Please check API keys, billing, and model access.",
        details: providerErrors,
      },
      { status: 500 },
    );
  } catch (error) {
    console.error("Analysis Server Error:", error);
    return NextResponse.json(
      { error: getErrorMessage(error) || "Failed to process skin analysis" },
      { status: 500 },
    );
  }
}
