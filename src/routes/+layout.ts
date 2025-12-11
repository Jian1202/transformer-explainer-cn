// src/routes/+layout.ts

// 禁用服务器端渲染，因为本项目依赖浏览器环境的 API (如 navigator, onnxruntime)
export const ssr = false;

// 启用预渲染（虽然禁用了 SSR，但仍需生成 index.html 入口文件）
export const prerender = true;