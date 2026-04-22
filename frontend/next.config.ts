import type { NextConfig } from "next";

// Pages serves the site at yardenmorad2003.github.io/weather-ml-v2/.
// On a local dev server (npm run dev) we skip the prefix so URLs stay short.
const isPagesBuild = process.env.GITHUB_ACTIONS === "true";
const basePath = isPagesBuild ? "/weather-ml-v2" : "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  assetPrefix: basePath ? `${basePath}/` : undefined,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
