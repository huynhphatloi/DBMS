/**
 * Hugging Face Inference API — Semantic similarity + NLI.
 * Free tier: ~30K chars/month for inference.
 */

const HF_SIM_URL =
  "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2";

export async function hfSemanticSimilarity(
  textA: string,
  textB: string
): Promise<number | null> {
  const token = process.env.HF_API_TOKEN;
  if (!token) return null;

  try {
    const res = await fetch(HF_SIM_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: { source_sentence: textA, sentences: [textB] },
      }),
    });
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
