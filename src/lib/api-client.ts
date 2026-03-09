export const api = {
  train: async (epochs?: number) => {
    const res = await fetch("http://localhost:8000/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: epochs || 10, config_path: "neurobridge.config.yaml" })
    });
    if (!res.ok) throw new Error("Train failed");
    return res.json();
  },
  evolve: async (generations?: number) => {
    const res = await fetch("http://localhost:8000/evolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ generations: generations || 1 })
    });
    if (!res.ok) throw new Error("Evolve failed");
    return res.json();
  },
  synthesize: async (sequence: string) => {
    const res = await fetch("http://localhost:8000/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence })
    });
    if (!res.ok) throw new Error("Synthesis failed");
    return res.json();
  }
};
