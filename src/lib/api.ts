export const api = {
  status: async () => {
    try {
      const response = await fetch("http://localhost:8000/status");
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    } catch (error) {
      console.error("Failed to fetch status:", error);
      throw error;
    }
  },
  train: async () => {
    try {
      const response = await fetch("http://localhost:8000/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ epochs: 10 }),
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    } catch (error) {
      console.error("Failed to start training:", error);
      throw error;
    }
  },
  evolve: async () => {
    try {
      const response = await fetch("http://localhost:8000/evolve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ generations: 100 }),
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    } catch (error) {
      console.error("Failed to start evolution:", error);
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
        body: JSON.stringify({ sequence }),
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    } catch (error) {
      console.error("Failed to synthesize:", error);
      throw error;
    }
  },
};
