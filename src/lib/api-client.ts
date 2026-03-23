const BASE_URL = "http://localhost:8000";

export const api = {
  getStatus: async () => {
    const res = await fetch(`${BASE_URL}/status`);
    if (!res.ok) throw new Error("Failed to fetch status");
    return res.json();
  },

  train: async (epochs: number = 10) => {
    const res = await fetch(`${BASE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs }),
    });
    if (!res.ok) throw new Error("Failed to start training");
    return res.json();
  },

  evolve: async (generations: number = 100) => {
    const res = await fetch(`${BASE_URL}/evolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ generations }),
    });
    if (!res.ok) throw new Error("Failed to start evolution");
    return res.json();
  },

  synthesize: async (sequence: string) => {
    const res = await fetch(`${BASE_URL}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    if (!res.ok) throw new Error("Failed to synthesize");
    return res.json();
  }
};
