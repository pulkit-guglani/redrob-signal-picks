import axios, { AxiosInstance } from "axios";
import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";

// ─── Config ──────────────────────────────────────────────────────────────────

const API_BASE_URL =
  process.env.API_BASE_URL || "https://api-signal.redrob.io/api";

let accessToken = process.env.ACCESS_TOKEN || "";
let refreshToken = process.env.REFRESH_TOKEN || "";

// Parse multiple API keys (comma-separated GEMINI_API_KEYS or single GEMINI_API_KEY)
const apiKeys: string[] = process.env.GEMINI_API_KEYS
  ? process.env.GEMINI_API_KEYS.split(",").map((k) => k.trim()).filter(Boolean)
  : process.env.GEMINI_API_KEY
    ? [process.env.GEMINI_API_KEY.trim()]
    : [];

if (!accessToken || !refreshToken || apiKeys.length === 0) {
  console.error("Missing required env vars: ACCESS_TOKEN, REFRESH_TOKEN, GEMINI_API_KEYS or GEMINI_API_KEY");
  process.exit(1);
}

// ─── API Client ───────────────────────────────────────────────────────────────

const api: AxiosInstance = axios.create({ baseURL: API_BASE_URL });

api.interceptors.request.use((config) => {
  config.headers.Authorization = `Bearer ${accessToken}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    if (error.response?.status === 401 && !error.config._retried) {
      error.config._retried = true;
      const refreshed = await refreshAccessToken();
      if (refreshed) {
        error.config.headers.Authorization = `Bearer ${accessToken}`;
        return api.request(error.config);
      }
    }
    return Promise.reject(error);
  }
);

async function refreshAccessToken(): Promise<boolean> {
  try {
    const res = await axios.post(
      `${API_BASE_URL}/v1/auth/refresh-token`,
      { refresh_token: refreshToken },
    );
    const data = res.data?.data;
    accessToken = data?.access_token || data?.data?.access_token;
    refreshToken = data?.refresh_token || data?.data?.refresh_token;
    console.log("Token refreshed successfully");
    return true;
  } catch (err: any) {
    console.error("Token refresh failed:", err.message);
    return false;
  }
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface UserBalance {
  availableBalance: number;
  visual?: {
    availableBalance?: number;
  };
}

interface EventOption {
  id: string;
  optionText: string;
}

interface Event {
  id: string;
  title: string;
  description?: string | null;
  status: string;
  minBetAmount?: number;
  maxBetAmount?: number;
  closesAt?: string | null;
  options: EventOption[];
  userPick?: unknown | null;
}

// ─── Fetch Balance ────────────────────────────────────────────────────────────

async function fetchUserBalance(): Promise<number> {
  const res = await api.get("/v1/users/balance");
  const payload = res.data?.data ?? res.data;
  const availableBalance = payload?.visual?.availableBalance ?? payload?.availableBalance ?? 0;
  return availableBalance;
}

// ─── Fetch Events ─────────────────────────────────────────────────────────────

async function fetchOpenEvents(): Promise<Event[]> {
  const allEvents: Event[] = [];
  let cursor: string | null = null;

  do {
    const res: any = await api.get("/v1/events", {
      params: { status: "OPENED", limit: 50, ...(cursor ? { cursor } : {}) },
    });
    const payload = res.data?.data;
    const page: Event[] = payload?.events || [];
    allEvents.push(...page);
    cursor = payload?.pagination?.nextCursor ?? null;
  } while (cursor);

  console.log(`Total events fetched: ${allEvents.length}`);

  const now = Date.now();
  const fiveHours = 5 * 60 * 60 * 1000;

  return allEvents.filter((e) => {
    if (e.userPick) {
      console.log(`  [skip] "${e.title}" — already picked`);
      return false;
    }
    if (!e.closesAt) {
      console.log(`  [skip] "${e.title}" — no closesAt`);
      return false;
    }
    const closesAt = new Date(e.closesAt).getTime();
    const minsLeft = Math.round((closesAt - now) / 60000);
    if (closesAt <= now) {
      console.log(`  [skip] "${e.title}" — already closed (closesAt: ${e.closesAt})`);
      return false;
    }
    if (closesAt - now > fiveHours) {
      console.log(`  [skip] "${e.title}" — closes in ${minsLeft} min (more than 3h)`);
      return false;
    }
    console.log(`  [pick] "${e.title}" — closes in ${minsLeft} min`);
    return true;
  });
}

// ─── AI Pick ──────────────────────────────────────────────────────────────────

// Model Priority List requested
const MODEL_PRIORITY_LIST: string[] = [
  "gemma-4-31b-it",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
];

// Cache GoogleGenerativeAI instances per API Key
const genAIInstances = new Map<string, GoogleGenerativeAI>();
function getGenAIInstance(key: string): GoogleGenerativeAI {
  if (!genAIInstances.has(key)) {
    genAIInstances.set(key, new GoogleGenerativeAI(key));
  }
  return genAIInstances.get(key)!;
}

// Sticky pointers: current Key index and current Model index
let currentKeyIndex = 0;
let currentModelIndex = 0;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

let last15RpmCallTimestamp = 0;

async function ensure15RpmPacing(modelName: string): Promise<void> {
  const is15RpmModel = !modelName.startsWith("gemma");
  if (!is15RpmModel) return;

  const now = Date.now();
  const elapsed = now - last15RpmCallTimestamp;
  const MIN_INTERVAL_MS = 4100; // 4.1s safety margin for 15 req/min (60s / 15 = 4s)

  if (elapsed < MIN_INTERVAL_MS && last15RpmCallTimestamp > 0) {
    const waitMs = MIN_INTERVAL_MS - elapsed;
    console.log(
      `  [Rate Limit Pacer] Waiting ${(waitMs / 1000).toFixed(1)}s for "${modelName}" (15 RPM limit)...`
    );
    await sleep(waitMs);
  }
  last15RpmCallTimestamp = Date.now();
}

function withTimeout<T>(promise: Promise<T>, ms: number, errorMessage: string): Promise<T> {
  let timer: NodeJS.Timeout;
  const timeoutPromise = new Promise<T>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(errorMessage));
    }, ms);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timer));
}

interface AIPickResult {
  optionId: string;
  confidence: number;
  modelUsed: string;
}

async function getAIPick(event: Event): Promise<AIPickResult | null> {
  const optionsList = event.options
    .map((o) => `[${o.id}] ${o.optionText}`)
    .join("\n");

  const prompt = `You are analyzing a prediction market event. Based on the question and options, pick the single most likely outcome.

Question: ${event.title}
${event.description ? `Context: ${event.description}` : ""}

Options:
${optionsList}

Reply with ONLY the raw option ID of your chosen option — no brackets, no explanation, nothing else. Example format: d0c14d45-9cf1-470f-84bb-ea8bccc39a40`;

  let keyIdx = currentKeyIndex;

  while (keyIdx < apiKeys.length) {
    const activeApiKey = apiKeys[keyIdx];
    const aiInstance = getGenAIInstance(activeApiKey);

    // Per-event model cascade starting from current non-rate-limited model index
    let modelIdx = keyIdx === currentKeyIndex ? currentModelIndex : 0;

    while (modelIdx < MODEL_PRIORITY_LIST.length) {
      const activeModelName = MODEL_PRIORITY_LIST[modelIdx];

      try {
        console.log(
          `  [AI] Querying Key #${keyIdx + 1} (${activeApiKey.slice(0, 6)}...) with model "${activeModelName}"`
        );

        const modelConfig: any = {
          model: activeModelName,
          generationConfig: {
            responseMimeType: "application/json",
            responseSchema: {
              type: SchemaType.OBJECT,
              properties: {
                reasoning: {
                  type: SchemaType.STRING,
                  description: "Step-by-step comparative evaluation of each option's probability.",
                },
                chosenOptionId: {
                  type: SchemaType.STRING,
                  description: "The exact ID of the option with the highest winning probability.",
                },
                confidence: {
                  type: SchemaType.NUMBER,
                  description: "Confidence score percentage (50 to 95) for the chosen option.",
                },
              },
              required: ["reasoning", "chosenOptionId", "confidence"],
            },
          },
        };

        // Enforce 15 RPM inter-request delay pacing for non-Gemma models
        await ensure15RpmPacing(activeModelName);

        const enableInternet = !activeModelName.includes("-lite");
        let result: any;

        if (enableInternet) {
          try {
            const modelWithTools = aiInstance.getGenerativeModel({
              ...modelConfig,
              tools: [{ googleSearch: {} } as any],
            });
            const timeoutMs = activeModelName.startsWith("gemma") ? 12000 : 25000;
            result = await withTimeout(
              modelWithTools.generateContent(prompt),
              timeoutMs,
              `"${activeModelName}" search query timed out after ${timeoutMs / 1000}s`
            );
          } catch (toolErr: any) {
            console.warn(
              `  [Internet Notice] "${activeModelName}" search tools failed (${toolErr.message}).`
            );
            if (activeModelName.startsWith("gemma")) {
              throw new Error(
                `[Internet Notice] "${activeModelName}" search tools failed/timed out (${toolErr.message}).`
              );
            }
            console.warn(`  Falling back to basic mode for "${activeModelName}"...`);
            const basicModel = aiInstance.getGenerativeModel(modelConfig);
            result = await withTimeout(
              basicModel.generateContent(prompt),
              20000,
              `"${activeModelName}" basic query timed out after 20s`
            );
          }
        } else {
          // Lite models run directly in basic mode without internet search tools
          const basicModel = aiInstance.getGenerativeModel(modelConfig);
          result = await withTimeout(
            basicModel.generateContent(prompt),
            20000,
            `"${activeModelName}" basic query timed out after 20s`
          );
        }

        const responseText = result.response.text();
        const parsed = JSON.parse(responseText);

        console.log(`  AI Reasoning (${activeModelName}): ${parsed.reasoning}`);
        const chosenOptionId = parsed.chosenOptionId?.trim();
        const match = event.options.find((o) => o.id === chosenOptionId);

        const parsedConfidence = Math.min(
          95,
          Math.max(50, Math.round(Number(parsed.confidence) || 70))
        );

        if (match) {
          return { optionId: match.id, confidence: parsedConfidence, modelUsed: activeModelName };
        }

        console.warn(`  AI returned unrecognized option ID: "${chosenOptionId}"`);
      } catch (err: any) {
        const isRateLimitErr =
          err.message?.includes("429") ||
          err.message?.includes("RESOURCE_EXHAUSTED") ||
          err.message?.toLowerCase().includes("rate limit");

        if (isRateLimitErr) {
          console.warn(
            `  [Rate Limit Backoff] "${activeModelName}" hit rate limit (429). Sleeping 10s before failover...`
          );
          await sleep(10000);

          currentModelIndex = modelIdx + 1;
          if (currentModelIndex >= MODEL_PRIORITY_LIST.length) {
            currentKeyIndex++;
            currentModelIndex = 0;
            keyIdx = currentKeyIndex;
            modelIdx = 0;
          } else {
            modelIdx = currentModelIndex;
          }
        } else {
          console.warn(
            `  [Event Model Fallback] "${activeModelName}" failed for this event (${err.message}). Trying next model...`
          );
          modelIdx++;
        }
      }
    }

    keyIdx++;
  }

  return null;
}

// ─── Reveal Picks ────────────────────────────────────────────────────────────

interface Pick {
  id: string;
  resultViewed: boolean;
  status: string;
  event: { title: string };
}

async function fetchUnrevealedPicks(): Promise<Pick[]> {
  const unrevealed: Pick[] = [];
  let cursor: string | null = null;
  let totalFetched = 0;
  let consecutiveViewed = 0;
  const MAX_PICKS = 200;
  const STOP_AFTER_VIEWED = 4;

  do {
    const res: any = await api.get("/v1/picks", {
      params: {
        status: ["WON", "LOST"],
        sortBy: "revealFirst",
        limit: 50,
        ...(cursor ? { cursor } : {}),
      },
      paramsSerializer: { indexes: null },
    });

    const payload = res.data?.data;
    const picks: Pick[] = payload?.picks || [];

    for (const pick of picks) {
      totalFetched++;
      if (!pick.resultViewed) {
        consecutiveViewed = 0;
        unrevealed.push(pick);
        console.log(`  [reveal] Queued: "${pick.event?.title}" — ${pick.status}`);
      } else {
        consecutiveViewed++;
        if (consecutiveViewed >= STOP_AFTER_VIEWED) {
          console.log(`  [reveal] Hit ${STOP_AFTER_VIEWED} consecutive viewed picks — stopping early at ${totalFetched} total`);
          return unrevealed;
        }
      }

      if (totalFetched >= MAX_PICKS) {
        console.log(`  [reveal] Reached ${MAX_PICKS} pick limit — stopping`);
        return unrevealed;
      }
    }

    cursor = payload?.pagination?.nextCursor ?? null;
  } while (cursor);

  console.log(`  [reveal] Done — fetched ${totalFetched}, unrevealed: ${unrevealed.length}`);
  return unrevealed;
}

async function revealPick(pickId: string): Promise<void> {
  console.log(`  [reveal] PATCH /v1/picks/${pickId}/result-viewed`);
  const res = await api.patch(`/v1/picks/${pickId}/result-viewed`);
  console.log(`  [reveal] Response status: ${res.status}`);
}

// ─── Place Pick ───────────────────────────────────────────────────────────────

async function placePick(
  eventId: string,
  optionId: string,
  amount: number,
  confidence: number = 70
): Promise<void> {
  await api.post("/v1/picks", {
    eventId,
    optionId,
    amount,
    confidence,
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\n[${new Date().toISOString()}] Starting auto-pick run...`);

  // ─── 1. Reveal settled picks ───────────────────────────────────────────────
  console.log("\nChecking for unrevealed results...");
  try {
    const unrevealed = await fetchUnrevealedPicks();
    console.log(`Found ${unrevealed.length} unrevealed pick(s)`);
    for (const pick of unrevealed) {
      try {
        await revealPick(pick.id);
        console.log(`  Revealed: "${pick.event.title}" — ${pick.status}`);
      } catch (err: any) {
        console.error(`  Error revealing pick ${pick.id}: ${err.response?.data?.message || err.message}`);
      }
    }
  } catch (err: any) {
    console.error("Failed to fetch unrevealed picks:", err.message);
  }

  // ─── 2. Check balance & place picks ─────────────────────────────────────────
  let availableBalance = 0;
  try {
    availableBalance = await fetchUserBalance();
    console.log(`User available balance: ${availableBalance}`);
  } catch (err: any) {
    console.error("Failed to fetch user balance:", err.message);
  }

  if (availableBalance <= 2000) {
    console.log(`Available balance (${availableBalance}) is <= 2000 — skipping open events check.`);
  } else {
    let events: Event[];
    try {
      events = await fetchOpenEvents();
    } catch (err: any) {
      console.error("Failed to fetch events:", err.message);
      process.exit(1);
    }

    console.log(`Found ${events.length} open event(s) without a pick`);

    for (const event of events) {
      console.log(`\nProcessing: "${event.title}"`);
      try {
        const aiResult = await getAIPick(event);
        if (!aiResult) {
          console.log(`  Skipped — AI could not determine an option`);
        } else {
          const minBetN = event.minBetAmount || 10;
          const maxBetN = event.maxBetAmount || minBetN;
          let amount: number;
          if (availableBalance > 600000 && maxBetN > minBetN) {
            // Use the quarter-of-max logic when balance > 6 lakh
            const quickAmounts = (() => {
              const quarter = Math.round(maxBetN / 4);
              const half = Math.round(maxBetN / 2);
              const amounts = [minBetN, quarter, half, maxBetN];
              return [...new Set(amounts)].sort((a, b) => a - b);
            })();
            amount = quickAmounts[1] ?? minBetN; // quarter is always index 1 (after minBet)
          } else {
            amount = minBetN;
          }
          await placePick(event.id, aiResult.optionId, amount, aiResult.confidence);
          const optionText = event.options.find((o) => o.id === aiResult.optionId)?.optionText;
          console.log(
            `  Picked: "${optionText}" (amount: ${amount}, confidence: ${aiResult.confidence}%, model: ${aiResult.modelUsed}, balance: ${availableBalance})`
          );
        }
      } catch (err: any) {
        const status = err?.status ?? err?.response?.status;
        if (status === 429) {
          console.warn(`  Skipped — Gemini rate limit hit, will retry next run`);
        } else {
          console.error(`  Error: ${err.response?.data?.message || err.message}`);
        }
      }
      await sleep(4000); // stay under 15 req/min free tier limit
    }
  }

  console.log(`\n[${new Date().toISOString()}] Run complete.`);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
