export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#14202b", mut: "#5b6773", line: "#d5dbe1", soft: "#f3f6f9",
        s0: "#dbe9f6", s1: "#9ecae1", s2: "#4292c6", s3: "#08306b",
        pass: "#1b6e3a", warn: "#8a5a00", fail: "#a4262c",
      },
      fontFamily: { sans: ["system-ui", "Segoe UI", "Roboto", "sans-serif"] },
    },
  },
  plugins: [],
};
