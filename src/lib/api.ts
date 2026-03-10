export const api = {
  train: async () => {
    const res = await fetch("http://localhost:8000/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ epochs: 10 }),
    });
    if (!res.ok) throw new Error("Training failed");
    return res.json();
  },

  evolve: async (generations: number = 100) => {
    const res = await fetch("http://localhost:8000/evolve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ generations }),
    });
    if (!res.ok) throw new Error("Evolution failed");
    return res.json();
  },

  synthesize: async (sequence: string) => {
    const res = await fetch("http://localhost:8000/synthesize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sequence }),
    });
    if (!res.ok) throw new Error("Synthesis failed");
    return res.json();
  },
};
