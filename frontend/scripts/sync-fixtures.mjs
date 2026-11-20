// Copy the synthetic report fixtures next to the app so /report/fixture-<name> works offline (gitignored copy).
import { cpSync, mkdirSync } from "node:fs";
mkdirSync("public/fixtures", { recursive: true });
cpSync("../tests/fixtures/reports", "public/fixtures", { recursive: true });
