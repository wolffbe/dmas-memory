# Distributed multi-agent system (DMAS) long context memory

Comparison of long-context vector vs graph memory in distributed LLM-based multi-agent systems using the LOCOMO dataset.

## Platform Notes

For docker.sock of telegraf monitoring

Linux: Set `DOCKER_GROUP_ID=$(getent group docker | cut -d: -f3)` in `.env`. Windows/macOS: Remove `group_add` from `monitoring/docker-compose.yml` and use `user: root` only.

## References

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
