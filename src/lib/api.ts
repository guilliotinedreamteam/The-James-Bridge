const BASE_URL = "http://localhost:8000";

export const api = {
  train: async () => {
    const response = await fetch(`${BASE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 10 }),
    });
    if (!response.ok) throw new Error("Training failed");
    return response.json();
  },
  synthesize: async (sequence: string) => {
    const response = await fetch(`${BASE_URL}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    if (!response.ok) throw new Error("Synthesis failed");
    return response.json();
  },
};
