/**
 * Hugging Face Inference API client (free tier).
 *
 * Priority:
 * 1. Custom fine-tuned model (HF_MODEL_ID) — if you trained one
 * 2. Public all-MiniLM-L6-v2 for similarity — always available
 *
 * Falls back gracefully if no token is set.
 */

const HF_BASE = "https://api-inference.huggingface.co/models";

/**
 * Call your custom fine-tuned grading model on HF Hub.
 * Returns predicted label + confidence, or null if unavailable.
 */
export async function hfGradeAnswer(
  question: string,
  referenceAnswer: string,
  studentAnswer: string
): Promise<{ label: string; score: number } | null> {
  const token = process.env.HF_API_TOKEN;
  const modelId = process.env.HF_MODEL_ID; // e.g. "your-username/asag-grading-deberta-v3"
  if (!token || !modelId) return null;

  try {
    const input = `${question} [SEP] ${referenceAnswer} [SEP] ${studentAnswer}`;
    const res = await fetch(`${HF_BASE}/${modelId}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: input, parameters: { truncation: true } }),
    });

    if (!res.ok) return null;
    const data = await res.json();

    // Response: [[{label, score}, ...]] sorted by score desc
    if (Array.isArray(data) && Array.isArray(data[0])) {
      const top = data[0][0];
      return { label: top.label, score: top.score };
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Semantic similarity via public all-MiniLM-L6-v2.
 */
export async function hfSemanticSimilarity(
  textA: string,
  textB: string
): Promise<number | null> {
  const token = process.env.HF_API_TOKEN;
  if (!token) return null;

  try {
    const res = await fetch(
      `${HF_BASE}/sentence-transformers/all-MiniLM-L6-v2`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: { source_sentence: textA, sentences: [textB] },
        }),
      }
    );

    if (!res.ok) return null;
    const data = await res.json();
    if (Array.isArray(data) && typeof data[0] === "number") {
      return Math.max(0, Math.min(1, data[0]));
    }
    return null;
  } catch {
    return null;
  }
}
