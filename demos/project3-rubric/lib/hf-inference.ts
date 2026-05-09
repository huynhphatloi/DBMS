/**
 * Hugging Face Inference API — NLI for rubric criterion matching.
 * Uses cross-encoder/nli-deberta-v3-xsmall (free tier friendly).
 */

const HF_NLI_URL =
  "https://api-inference.huggingface.co/models/cross-encoder/nli-deberta-v3-xsmall";

export async function hfNliCheck(
  premise: string,
  hypothesis: string
): Promise<{ label: string; score: number } | null> {
  const token = process.env.HF_API_TOKEN;
  if (!token) return null;

  try {
    const res = await fetch(HF_NLI_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: `${premise} [SEP] ${hypothesis}`,
      }),
    });

    if (!res.ok) return null;
    const data = await res.json();
    if (Array.isArray(data) && Array.isArray(data[0])) {
      const sorted = data[0].sort(
        (a: { score: number }, b: { score: number }) => b.score - a.score
      );
      return { label: sorted[0].label, score: sorted[0].score };
    }
    return null;
  } catch {
    return null;
  }
}
