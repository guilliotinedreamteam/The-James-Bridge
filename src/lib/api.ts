const BASE_URL = "http://localhost:8000";

export const api = {
  async train() {
    const response = await fetch(`${BASE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 10 }),
    });
    if (!response.ok) {
      throw new Error("Failed to start training");
    }
    return response.json();
  },

  async synthesize(sequence: string) {
    const response = await fetch(`${BASE_URL}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to synthesize");
    }
    return response.json();
  }
};
