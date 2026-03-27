const API_BASE_URL = "http://localhost:8000";

export const api = {
  train: async () => {
    const res = await fetch(`${API_BASE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 10 }),
    });
    if (!res.ok) throw new Error("Failed to start training");
    return res.json();
  },

  evolve: async () => {
    const res = await fetch(`${API_BASE_URL}/evolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ generations: 100 }),
    });
    if (!res.ok) throw new Error("Failed to start evolution");
    return res.json();
  },

  synthesize: async (sequence: string) => {
    const res = await fetch(`${API_BASE_URL}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    if (!res.ok) throw new Error("Synthesis failed");
    return res.json();
  },
};
