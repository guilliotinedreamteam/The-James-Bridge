const BASE_URL = "http://localhost:8000";

export const api = {
  getStatus: async () => {
    const response = await fetch(`${BASE_URL}/status`);
    if (!response.ok) throw new Error("Failed to fetch status");
    return response.json();
  },

  train: async (epochs: number = 10) => {
    const response = await fetch(`${BASE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs }),
    });
    if (!response.ok) throw new Error("Failed to start training");
    return response.json();
  },

  synthesize: async (sequence: string) => {
    const response = await fetch(`${BASE_URL}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    if (!response.ok) throw new Error("Failed to synthesize");
    return response.json();
  },
};
