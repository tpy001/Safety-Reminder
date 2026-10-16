# Verification record (2026-10-16)

Validation used Python 3.10, PyTorch 2.4.0+cu124, Transformers 4.46.0, and available CUDA GPUs. These checks establish functionality, not completion of new 20/50-epoch experiments.

## Core scope and portable paths

- The runtime package contains model loading, SAPT, training data selection, soft prompt training, inference, optional scoring, and input/output conversion. Historical one-off scripts are archived and are not runtime dependencies.
- Core source, examples, and README contain no original server/user absolute paths. Model, data, output, and device are configurable. Bundled resources resolve relative to the installed package.
- A wheel was installed into a separate directory and used from another working directory for the GPU checks below. Models and datasets were provisioned locally for that directory; no old project code was imported.

## Completed checks

| Check | Observed result |
| --- | --- |
| CPU unit tests | 20 passed: SAPT timing, threshold, stopping, JSONL, MM-Vet, training gradients, sampling, and portable paths |
| Real LLaVA training | Two harmful and two benign examples; one optimizer update; only 4 x 4096 soft prompt parameters changed; no base-model gradients |
| Checkpoint reload | New weights loaded successfully for text and image inference |
| Installed-wheel inference from another directory | Two examples each for baseline, historical SAPT, and newly trained SAPT; two local Parquet SAPT examples |
| Local scoring | Two examples scored; no unknown labels |
| MM-Vet | All 218 inputs prepared with relative images and no reference answers; historical full baseline/SAPT predictions each exported to 218 official IDs; automatic CLI export also unit tested |
| Fixed subset IDs | All FigStep 350, MMSafety 80, and SPA-VL harm 53 IDs resolved at the pinned dataset revision |
| Dependency consistency | `pip check` reported no broken requirements |

The real training update had LM loss 1.092293031754032, classification loss 1.1820395927061327, and total loss 1.1513950113893388. The relocated run changed the prompt by a maximum absolute value of 0.00010007619857788086. These values verify training, saving, and reloading, not convergence or paper scores.

## Historical compatibility

- Paper equations (10)-(13) were checked explicitly: `lm_loss` is text generation L_gen, `cls_loss` supervises the final soft prompt hidden state through the frozen PCA classifier, and the total is `L_gen + 0.05 * L_cls`. Both components independently produce finite, nonzero soft prompt gradients; classifier tensors remain frozen.
- Default 256-per-pool selection reproduces all 512 historical sample IDs and their shuffled order. Eight real examples match the old training text and label masks. The full-pool mode also produces 256 harmful and 256 benign examples.
- Three baseline cases (text, image, and safety question) and six SAPT cases (natural and forced triggers) were compared under identical dependencies, inputs, and weights. Tokens, check positions, and answers match; classifier probabilities were checked with 1e-6 tolerance. Forced triggers cover control flow only.
- All three safetensors shards and 11 model configuration/processor files match the pinned upstream revision documented in README.
- The new standalone scorer re-scored 53 historical SPA-VL answers: baseline unsafe 19/53 (35.85%), SAPT 0/53, with zero unknown labels in both. This reuses existing generations to check scorer migration; it is not a fresh full-paper experiment.

Training accumulates one-example forwards while retaining token-mean LM and example-mean classification objectives. Hardware, arithmetic, and random prefix ordering can change trajectories. The archived full SPA-VL baseline differs from the paper itself; subset results cannot prove full reproduction.

## Records and distribution

Detailed logs and audit JSON are retained in the development workspace under `outputs/project_cleanup/`, outside the runtime package. Test files remain in that workspace and are excluded from this branch. The results above record checks completed during development.

On a new machine, follow README to install dependencies and prepare the pinned model/data, then run a short training check. The GPU checks do not establish compatibility with every GPU, driver, or operating system.
