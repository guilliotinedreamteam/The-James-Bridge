// src/lib/api.ts

export const api = {
  train: async () => {
    try {
      const response = await fetch("http://localhost:8000/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          config_path: "neurobridge.config.yaml",
          epochs: 10,
        }),
      });

      if (!response.ok) {
        throw new Error(`Training request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in api.train:", error);
      throw error;
    }
  },

  synthesize: async (sequence: string) => {
    try {
      const response = await fetch("http://localhost:8000/synthesize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sequence,
        }),
      });

      if (!response.ok) {
        throw new Error(`Synthesis request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in api.synthesize:", error);
      throw error;
    }
  },
};
