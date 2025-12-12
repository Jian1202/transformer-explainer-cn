# Transformer Explainer（中文汉化版）

> 基于原版 [Transformer Explainer](http://poloclub.github.io/transformer-explainer) 的中文本地化版本，完整保留原版功能与交互逻辑，仅对界面文本、标签和提示信息进行汉化，方便中文用户学习与教学使用。

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2408.04619-red)](https://arxiv.org/abs/2408.04619)

<a href="https://youtu.be/TFUc41G2ikY" target="_blank">
  <img width="100%" src="https://github.com/user-attachments/assets/0a4d8888-6555-4df5-bc71-77f1299115c3" alt="Transformer Explainer 演示视频">
</a>

## ✨ 项目特色

- **完整中文化界面**：所有操作界面、标签、提示和说明均已汉化
- **功能无损**：保持与原版完全一致的模型交互、可视化与分析功能
- **开箱即用**：支持本地快速部署，无需额外配置
- **教学友好**：适合中文环境下的Transformer模型教学与演示

## 📋 版本说明

- 本仓库为原版 [Transformer Explainer](https://github.com/poloclub/transformer-explainer) 的中文汉化版本
- 仅对界面文本进行本地化处理，核心功能、交互逻辑与可视化组件保持不变
- 遵循原项目的 MIT 开源协议
- **原版地址**：[http://poloclub.github.io/transformer-explainer](http://poloclub.github.io/transformer-explainer)

## 🚀 快速开始

### 环境要求

- Node.js v20 或更高版本
- NPM v10 或更高版本

### 安装与运行

1. **克隆仓库**
   ```bash
   git clone https://github.com/Jian1202/transformer-explainer-cn.git
   cd transformer-explainer-cn
   ```

2. **安装依赖**
   ```bash
   npm install
   ```

3. **启动开发服务器**
   ```bash
   npm run dev
   ```

4. **访问应用**
   打开浏览器，访问 [http://localhost:5173](http://localhost:5173) 即可体验中文界面。

## 🌐 在线体验

访问中文汉化版在线演示或直接访问原版英文界面：
- **中文版**：[https://jian1202.github.io/transformer-explainer-cn](https://jian1202.github.io/transformer-explainer-cn)
- **英文原版**：[http://poloclub.github.io/transformer-explainer](http://poloclub.github.io/transformer-explainer)

## 📖 功能特性

- **交互式学习**：通过可视化界面深入理解Transformer模型的工作原理
- **注意力机制可视化**：直观展示自注意力机制的权重分布
- **文本生成过程分析**：逐步分解文本生成过程，揭示模型决策逻辑
- **多层级解释**：从词元级别到层级的全方位模型解释
- **实时交互**：支持参数调整与实时结果反馈

## 🏗️ 项目结构

```
transformer-explainer-cn/
├── src/                    # 源代码目录
│   ├── locales/           # 多语言文件（含中文翻译）
│   ├── components/        # 组件文件
│   └── ...
├── public/                # 静态资源
├── package.json           # 项目配置
├── README.md              # 说明文档（本文件）
└── ...
```

## 📄 引用与致谢

### 原项目团队
Transformer Explainer 由佐治亚理工学院团队开发：
- [Aeree Cho](https://aereeeee.github.io/)
- [Grace C. Kim](https://www.linkedin.com/in/chaeyeonggracekim/)
- [Alexander Karpekov](https://alexkarpekov.com/)
- [Alec Helbling](https://alechelbling.com/)
- [Jay Wang](https://zijie.wang/)
- [Seongmin Lee](https://seongmin.xyz/)
- [Benjamin Hoover](https://bhoov.com/)
- [Polo Chau](https://poloclub.github.io/polochau/)

### 引用格式
如需在研究中引用本项目，请使用以下 BibTeX 格式：

```bibtex
@article{cho2024transformer,
  title = {Transformer Explainer: Interactive Learning of Text-Generative Models},
  shorttitle = {Transformer Explainer},
  author = {Cho, Aeree and Kim, Grace C. and Karpekov, Alexander and Helbling, Alec and Wang, Zijie J. and Lee, Seongmin and Hoover, Benjamin and Chau, Duen Horng},
  journal = {IEEE VIS Poster},
  year = {2024}
}
```

## 📜 开源协议

本项目基于 [MIT License](LICENSE) 开源。

## 🤝 贡献与反馈

- 如有翻译问题或改进建议，欢迎提交 Issue 或 Pull Request
- 中文相关问题可联系本仓库维护者
- 功能性问题建议反馈至原仓库：[原项目 Issues](https://github.com/poloclub/transformer-explainer/issues/new/choose)

## 🔗 相关项目推荐

- [**Diffusion Explainer**](https://poloclub.github.io/diffusion-explainer) - 可视化学习Stable Diffusion如何将文本提示转化为图像
- [**CNN Explainer**](https://poloclub.github.io/cnn-explainer) - 卷积神经网络交互式学习工具
- [**GAN Lab**](https://poloclub.github.io/ganlab) - 浏览器中的生成对抗网络实验平台

## 📧 联系信息

- **原项目联系人**：[Aeree Cho](https://aereeeee.github.io/)
- **中文版维护**：本仓库维护者

---

*本中文汉化版仅用于学习与研究目的，所有模型与算法版权归原项目所有。*

--- 

**温馨提示**：使用过程中如遇技术问题，建议优先参考原项目文档与 Issues。
