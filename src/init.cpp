// Auto-generated registration for real (libtorch) build
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern "C" SEXP _tinytorch_C_tensor_requires_grad_(SEXP self_sexpSEXP, SEXP requires_gradSEXP);
extern "C" SEXP _tinytorch_C_tensor_grad(SEXP self_sexpSEXP);
extern "C" SEXP _tinytorch_C_tensor_backward(SEXP self_sexpSEXP, SEXP gradient_sexpSEXP, SEXP retain_graphSEXP, SEXP create_graphSEXP);
extern "C" SEXP _tinytorch_C_autograd_set_grad_mode(SEXP enabledSEXP);
extern "C" SEXP _tinytorch_C_autograd_is_enabled();
extern "C" SEXP _tinytorch_C_autograd_grad(SEXP outputs_sexpSEXP, SEXP inputs_sexpSEXP, SEXP grad_outputs_sexpSEXP, SEXP retain_graphSEXP, SEXP create_graphSEXP, SEXP allow_unusedSEXP);
extern "C" SEXP _tinytorch_C_tensor_is_leaf(SEXP self_sexpSEXP);
extern "C" SEXP _tinytorch_C_tensor_retain_grad(SEXP self_sexpSEXP);
extern "C" SEXP _tinytorch_C_as_array(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_tensor_shape(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_tensor_dtype(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_tensor_device(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_tensor_ndim(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_tensor_numel(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_tensor_requires_grad(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_tensor_print(SEXP selfSEXP);
extern "C" SEXP _tinytorch_cpp_fused_relu(SEXP input_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_fused_relu_sigmoid(SEXP input_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_fused_relu_sigmoid_tanh(SEXP input_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_fused_silu(SEXP input_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_fused_gelu(SEXP input_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_fused_sincos(SEXP input_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_fused_softcap(SEXP input_sexpSEXP, SEXP cap_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_fused_rmsnorm(SEXP input_sexpSEXP, SEXP weight_sexpSEXP, SEXP eps_sexpSEXP);
extern "C" SEXP _tinytorch_cpp_tensor_shapes_key(SEXP tensor_listSEXP);
extern "C" SEXP _tinytorch_C_torch_rename_(SEXP selfSEXP, SEXP names_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_rename(SEXP selfSEXP, SEXP names_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_align_to(SEXP selfSEXP, SEXP names_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_align_as(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_align_tensors(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sym_constrain_range(SEXP size_sexpSEXP, SEXP minSEXP, SEXP maxSEXP);
extern "C" SEXP _tinytorch_C_torch_sym_constrain_range_for_size(SEXP size_sexpSEXP, SEXP minSEXP, SEXP maxSEXP);
extern "C" SEXP _tinytorch_C_torch_refine_names(SEXP selfSEXP, SEXP names_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_native_dropout(SEXP inputSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_dropout(SEXP inputSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_dropout_(SEXP selfSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_feature_dropout(SEXP inputSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_feature_dropout_(SEXP selfSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_alpha_dropout(SEXP inputSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_alpha_dropout_(SEXP selfSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_feature_alpha_dropout(SEXP inputSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_feature_alpha_dropout_(SEXP selfSEXP, SEXP pSEXP, SEXP trainSEXP);
extern "C" SEXP _tinytorch_C_torch_abs_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_absolute(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_absolute_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_angle(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_view_as_complex(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sgn(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sgn_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_chalf(SEXP selfSEXP, SEXP memory_formatSEXP);
extern "C" SEXP _tinytorch_C_torch_conj_physical(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_conj_physical_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_resolve_conj(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_resolve_neg(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_acos(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_acos_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arccos(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arccos_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_avg_pool1d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP ceil_modeSEXP, SEXP count_include_padSEXP);
extern "C" SEXP _tinytorch_C_torch_adaptive_avg_pool1d(SEXP selfSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_adaptive_max_pool1d(SEXP selfSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_add_(SEXP selfSEXP, SEXP otherSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addmv(SEXP selfSEXP, SEXP matSEXP, SEXP vecSEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addmv_(SEXP selfSEXP, SEXP matSEXP, SEXP vecSEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addr(SEXP selfSEXP, SEXP vec1SEXP, SEXP vec2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addr_(SEXP selfSEXP, SEXP vec1SEXP, SEXP vec2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_affine_grid_generator(SEXP thetaSEXP, SEXP size_sexpSEXP, SEXP align_cornersSEXP);
extern "C" SEXP _tinytorch_C_torch_all(SEXP selfSEXP, SEXP dimSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_any(SEXP selfSEXP, SEXP dimSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_acosh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_acosh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arccosh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arccosh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_asinh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_asinh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arcsinh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arcsinh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_atanh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_atanh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arctanh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arctanh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_as_strided(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP stride_sexpSEXP, SEXP storage_offsetSEXP);
extern "C" SEXP _tinytorch_C_torch_as_strided_(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP stride_sexpSEXP, SEXP storage_offsetSEXP);
extern "C" SEXP _tinytorch_C_torch_asin(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_asin_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arcsin(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arcsin_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_atan(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_atan_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arctan(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_arctan_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_atleast_1d(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_atleast_2d(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_atleast_3d(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_baddbmm(SEXP selfSEXP, SEXP batch1SEXP, SEXP batch2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_baddbmm_(SEXP selfSEXP, SEXP batch1SEXP, SEXP batch2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bartlett_window(SEXP window_lengthSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_batch_norm(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP meanSEXP, SEXP varSEXP, SEXP epsSEXP, SEXP output_scaleSEXP, SEXP output_zero_pointSEXP);
extern "C" SEXP _tinytorch_C_torch_bernoulli(SEXP selfSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_bernoulli_(SEXP selfSEXP, SEXP pSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_bilinear(SEXP input1SEXP, SEXP input2SEXP, SEXP weightSEXP, SEXP biasSEXP);
extern "C" SEXP _tinytorch_C_torch_binary_cross_entropy(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_binary_cross_entropy_with_logits(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP pos_weightSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_bincount(SEXP selfSEXP, SEXP weightsSEXP, SEXP minlengthSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_not(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_not_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_copysign(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_copysign_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_not_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_xor(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_xor_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_and(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_and_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_or(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_or_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_blackman_window(SEXP window_lengthSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_broadcast_tensors(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_broadcast_to(SEXP selfSEXP, SEXP size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_concat(SEXP tensors_sexpSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_concatenate(SEXP tensors_sexpSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_block_diag(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_ceil_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_chain_matmul(SEXP matrices_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_unsafe_chunk(SEXP selfSEXP, SEXP chunksSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_chunk(SEXP selfSEXP, SEXP chunksSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_tensor_split(SEXP selfSEXP, SEXP sectionsSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_clamp_(SEXP selfSEXP, SEXP minSEXP, SEXP maxSEXP);
extern "C" SEXP _tinytorch_C_torch_clamp_max(SEXP selfSEXP, SEXP max_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_clamp_max_(SEXP selfSEXP, SEXP max_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_clamp_min(SEXP selfSEXP, SEXP min_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_clamp_min_(SEXP selfSEXP, SEXP min_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_clip(SEXP selfSEXP, SEXP minSEXP, SEXP maxSEXP);
extern "C" SEXP _tinytorch_C_torch_clip_(SEXP selfSEXP, SEXP minSEXP, SEXP maxSEXP);
extern "C" SEXP _tinytorch_C_torch_constant_pad_nd(SEXP selfSEXP, SEXP pad_sexpSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_convolution(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP transposedSEXP, SEXP output_padding_sexpSEXP, SEXP groupsSEXP);
extern "C" SEXP _tinytorch_C_torch_convolution_overrideable(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP transposedSEXP, SEXP output_padding_sexpSEXP, SEXP groupsSEXP);
extern "C" SEXP _tinytorch_C_torch_conv3d(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP groupsSEXP);
extern "C" SEXP _tinytorch_C_torch_conv_tbc(SEXP selfSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP padSEXP);
extern "C" SEXP _tinytorch_C_torch_conv_transpose2d(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP output_padding_sexpSEXP, SEXP groupsSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_conv_transpose3d(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP output_padding_sexpSEXP, SEXP groupsSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_copy(SEXP selfSEXP, SEXP srcSEXP, SEXP non_blockingSEXP);
extern "C" SEXP _tinytorch_C_torch_cos_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_cosh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_cosh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_cosine_embedding_loss(SEXP input1SEXP, SEXP input2SEXP, SEXP targetSEXP, SEXP marginSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_count_nonzero(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_cov(SEXP selfSEXP, SEXP correctionSEXP, SEXP fweightsSEXP, SEXP aweightsSEXP);
extern "C" SEXP _tinytorch_C_torch_corrcoef(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_cummax(SEXP selfSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_cummin(SEXP selfSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_cumprod(SEXP selfSEXP, SEXP dimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_cumprod_(SEXP selfSEXP, SEXP dimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_cumsum_(SEXP selfSEXP, SEXP dimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_cumulative_trapezoid(SEXP ySEXP, SEXP xSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_ctc_loss(SEXP log_probsSEXP, SEXP targetsSEXP, SEXP input_lengths_sexpSEXP, SEXP target_lengths_sexpSEXP, SEXP blankSEXP, SEXP reductionSEXP, SEXP zero_infinitySEXP);
extern "C" SEXP _tinytorch_C_torch_diag_embed(SEXP selfSEXP, SEXP offsetSEXP, SEXP dim1SEXP, SEXP dim2SEXP);
extern "C" SEXP _tinytorch_C_torch_diagflat(SEXP selfSEXP, SEXP offsetSEXP);
extern "C" SEXP _tinytorch_C_torch_diagonal(SEXP selfSEXP, SEXP offsetSEXP, SEXP dim1SEXP, SEXP dim2SEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_diagonal(SEXP ASEXP, SEXP offsetSEXP, SEXP dim1SEXP, SEXP dim2SEXP);
extern "C" SEXP _tinytorch_C_torch_fill_diagonal_(SEXP selfSEXP, SEXP fill_value_sexpSEXP, SEXP wrapSEXP);
extern "C" SEXP _tinytorch_C_torch_diff(SEXP selfSEXP, SEXP nSEXP, SEXP dimSEXP, SEXP prependSEXP, SEXP appendSEXP);
extern "C" SEXP _tinytorch_C_torch_gradient(SEXP selfSEXP, SEXP spacingSEXP, SEXP dimSEXP, SEXP edge_orderSEXP);
extern "C" SEXP _tinytorch_C_torch_div_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_divide(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_divide_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_true_divide(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_true_divide_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_dot(SEXP selfSEXP, SEXP tensorSEXP);
extern "C" SEXP _tinytorch_C_torch_vdot(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_einsum(SEXP equationSEXP, SEXP tensors_sexpSEXP, SEXP path_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_embedding_renorm_(SEXP selfSEXP, SEXP indicesSEXP, SEXP max_normSEXP, SEXP norm_typeSEXP);
extern "C" SEXP _tinytorch_C_torch_row_stack(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_embedding_bag(SEXP weightSEXP, SEXP indicesSEXP, SEXP offsetsSEXP, SEXP scale_grad_by_freqSEXP, SEXP modeSEXP, SEXP sparseSEXP, SEXP per_sample_weightsSEXP, SEXP include_last_offsetSEXP);
extern "C" SEXP _tinytorch_C_torch_empty_permuted(SEXP size_sexpSEXP, SEXP physical_layout_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_new_empty(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_new_empty_strided(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP stride_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_new_full(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP fill_value_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_new_zeros(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_new_ones(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_resize_(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP memory_formatSEXP);
extern "C" SEXP _tinytorch_C_torch_empty_quantized(SEXP size_sexpSEXP, SEXP qtensorSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP, SEXP memory_formatSEXP);
extern "C" SEXP _tinytorch_C_torch_empty_strided(SEXP size_sexpSEXP, SEXP stride_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_erf(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_erf_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_erfc(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_erfc_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_exp_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_exp2(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_exp2_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_expm1(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_expm1_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_expand_as(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_eye(SEXP nSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_unflatten(SEXP selfSEXP, SEXP dimSEXP, SEXP sizes_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_fill(SEXP selfSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_floor_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_floor_divide_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_frac(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_frac_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_full_like(SEXP selfSEXP, SEXP fill_value_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP, SEXP memory_formatSEXP);
extern "C" SEXP _tinytorch_C_torch_from_file(SEXP filenameSEXP, SEXP sharedSEXP, SEXP sizeSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_gcd(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_gcd_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_lcm(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_lcm_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_grid_sampler(SEXP inputSEXP, SEXP gridSEXP, SEXP interpolation_modeSEXP, SEXP padding_modeSEXP, SEXP align_cornersSEXP);
extern "C" SEXP _tinytorch_C_torch_grid_sampler_2d(SEXP inputSEXP, SEXP gridSEXP, SEXP interpolation_modeSEXP, SEXP padding_modeSEXP, SEXP align_cornersSEXP);
extern "C" SEXP _tinytorch_C_torch_grid_sampler_3d(SEXP inputSEXP, SEXP gridSEXP, SEXP interpolation_modeSEXP, SEXP padding_modeSEXP, SEXP align_cornersSEXP);
extern "C" SEXP _tinytorch_C_torch_hamming_window(SEXP window_lengthSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_kaiser_window(SEXP window_lengthSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_hinge_embedding_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP marginSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_group_norm(SEXP inputSEXP, SEXP num_groupsSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP epsSEXP, SEXP cudnn_enabledSEXP);
extern "C" SEXP _tinytorch_C_torch_native_group_norm(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP NSEXP, SEXP CSEXP, SEXP HxWSEXP, SEXP groupSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_index_copy_(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP sourceSEXP);
extern "C" SEXP _tinytorch_C_torch_index_put_(SEXP selfSEXP, SEXP indices_sexpSEXP, SEXP valuesSEXP, SEXP accumulateSEXP);
extern "C" SEXP _tinytorch_C_torch_instance_norm(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP running_meanSEXP, SEXP running_varSEXP, SEXP use_input_statsSEXP, SEXP momentumSEXP, SEXP epsSEXP, SEXP cudnn_enabledSEXP);
extern "C" SEXP _tinytorch_C_torch_isclose(SEXP selfSEXP, SEXP otherSEXP, SEXP rtolSEXP, SEXP atolSEXP, SEXP equal_nanSEXP);
extern "C" SEXP _tinytorch_C_torch_isin(SEXP elementsSEXP, SEXP test_elementsSEXP, SEXP assume_uniqueSEXP, SEXP invertSEXP);
extern "C" SEXP _tinytorch_C_torch_isnan(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_is_distributed(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_isreal(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_is_nonzero(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_is_same_size(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_kl_div(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP, SEXP log_targetSEXP);
extern "C" SEXP _tinytorch_C_torch_kron(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_kthvalue(SEXP selfSEXP, SEXP kSEXP, SEXP dimSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_native_layer_norm(SEXP inputSEXP, SEXP normalized_shape_sexpSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_rms_norm(SEXP inputSEXP, SEXP normalized_shape_sexpSEXP, SEXP weightSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_nan_to_num(SEXP selfSEXP, SEXP nanSEXP, SEXP posinfSEXP, SEXP neginfSEXP);
extern "C" SEXP _tinytorch_C_torch_nan_to_num_(SEXP selfSEXP, SEXP nanSEXP, SEXP posinfSEXP, SEXP neginfSEXP);
extern "C" SEXP _tinytorch_C_torch_ldexp(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_ldexp_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_log_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log10_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log1p(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log1p_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log2_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_logaddexp(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logaddexp2(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_xlogy(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_xlogy_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_logspace(SEXP start_sexpSEXP, SEXP end_sexpSEXP, SEXP stepsSEXP, SEXP baseSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_logcumsumexp(SEXP selfSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_logsumexp(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_margin_ranking_loss(SEXP input1SEXP, SEXP input2SEXP, SEXP targetSEXP, SEXP marginSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_matrix_power(SEXP selfSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_matrix_exp(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_aminmax(SEXP selfSEXP, SEXP dimSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_amax(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_max_pool1d_with_indices(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_max_pool1d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_max_pool2d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_max_pool1d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_max_pool2d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_max_pool3d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_max_pool3d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_nanmean(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_median(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_nanmedian(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_amin(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_mode(SEXP selfSEXP, SEXP dimSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_mul_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_multiply(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_multiply_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_mv(SEXP selfSEXP, SEXP vecSEXP);
extern "C" SEXP _tinytorch_C_torch_mvlgamma(SEXP selfSEXP, SEXP pSEXP);
extern "C" SEXP _tinytorch_C_torch_mvlgamma_(SEXP selfSEXP, SEXP pSEXP);
extern "C" SEXP _tinytorch_C_torch_native_batch_norm(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP running_meanSEXP, SEXP running_varSEXP, SEXP trainingSEXP, SEXP momentumSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_batch_norm_stats(SEXP inputSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_batch_norm_elemt(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP meanSEXP, SEXP invstdSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_batch_norm_gather_stats(SEXP inputSEXP, SEXP meanSEXP, SEXP invstdSEXP, SEXP running_meanSEXP, SEXP running_varSEXP, SEXP momentumSEXP, SEXP epsSEXP, SEXP countSEXP);
extern "C" SEXP _tinytorch_C_torch_batch_norm_gather_stats_with_counts(SEXP inputSEXP, SEXP meanSEXP, SEXP invstdSEXP, SEXP running_meanSEXP, SEXP running_varSEXP, SEXP momentumSEXP, SEXP epsSEXP, SEXP countsSEXP);
extern "C" SEXP _tinytorch_C_torch_batch_norm_update_stats(SEXP inputSEXP, SEXP running_meanSEXP, SEXP running_varSEXP, SEXP momentumSEXP);
extern "C" SEXP _tinytorch_C_torch_is_vulkan_available();
extern "C" SEXP _tinytorch_C_torch_pairwise_distance(SEXP x1SEXP, SEXP x2SEXP, SEXP pSEXP, SEXP epsSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_cdist(SEXP x1SEXP, SEXP x2SEXP, SEXP pSEXP, SEXP compute_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_pdist(SEXP selfSEXP, SEXP pSEXP);
extern "C" SEXP _tinytorch_C_torch_cosine_similarity(SEXP x1SEXP, SEXP x2SEXP, SEXP dimSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_movedim(SEXP selfSEXP, SEXP source_sexpSEXP, SEXP destination_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_moveaxis(SEXP selfSEXP, SEXP source_sexpSEXP, SEXP destination_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_numpy_T(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_matrix_H(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_mT(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_mH(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_adjoint(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_pixel_shuffle(SEXP selfSEXP, SEXP upscale_factorSEXP);
extern "C" SEXP _tinytorch_C_torch_pixel_unshuffle(SEXP selfSEXP, SEXP downscale_factorSEXP);
extern "C" SEXP _tinytorch_C_torch_channel_shuffle(SEXP selfSEXP, SEXP groupsSEXP);
extern "C" SEXP _tinytorch_C_torch_native_channel_shuffle(SEXP selfSEXP, SEXP groupsSEXP);
extern "C" SEXP _tinytorch_C_torch_is_pinned(SEXP selfSEXP, SEXP deviceSEXP);
extern "C" SEXP _tinytorch_C_torch_pin_memory(SEXP selfSEXP, SEXP deviceSEXP);
extern "C" SEXP _tinytorch_C_torch_pinverse(SEXP selfSEXP, SEXP rcondSEXP);
extern "C" SEXP _tinytorch_C_torch_poisson_nll_loss(SEXP inputSEXP, SEXP targetSEXP, SEXP log_inputSEXP, SEXP fullSEXP, SEXP epsSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_rad2deg(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_rad2deg_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_deg2rad(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_deg2rad_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_scalar_tensor(SEXP s_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_rand(SEXP size_sexpSEXP, SEXP names_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_rand_like(SEXP selfSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP, SEXP memory_formatSEXP);
extern "C" SEXP _tinytorch_C_torch_randint(SEXP highSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_randint_like(SEXP selfSEXP, SEXP highSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP, SEXP memory_formatSEXP);
extern "C" SEXP _tinytorch_C_torch_randperm(SEXP nSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_range(SEXP start_sexpSEXP, SEXP end_sexpSEXP, SEXP step_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_ravel(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_reciprocal(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_reciprocal_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_neg_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_negative(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_negative_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_reshape_as(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_round_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_rrelu(SEXP selfSEXP, SEXP lower_sexpSEXP, SEXP upper_sexpSEXP, SEXP trainingSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_rrelu_(SEXP selfSEXP, SEXP lower_sexpSEXP, SEXP upper_sexpSEXP, SEXP trainingSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_relu_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_relu6(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_relu6_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_prelu(SEXP selfSEXP, SEXP weightSEXP);
extern "C" SEXP _tinytorch_C_torch_gelu_(SEXP selfSEXP, SEXP approximateSEXP);
extern "C" SEXP _tinytorch_C_torch_hardshrink(SEXP selfSEXP, SEXP lambd_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_rsqrt_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_select(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP);
extern "C" SEXP _tinytorch_C_torch_selu(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_selu_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_celu(SEXP selfSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_celu_(SEXP selfSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_silu_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_mish(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_mish_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sigmoid_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_logit(SEXP selfSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_logit_(SEXP selfSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_sin_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sinc(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sinc_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sinh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sinh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_detach_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_slice(SEXP selfSEXP, SEXP dimSEXP, SEXP startSEXP, SEXP endSEXP, SEXP stepSEXP);
extern "C" SEXP _tinytorch_C_torch_slice_inverse(SEXP selfSEXP, SEXP srcSEXP, SEXP dimSEXP, SEXP startSEXP, SEXP endSEXP, SEXP stepSEXP);
extern "C" SEXP _tinytorch_C_torch_slice_scatter(SEXP selfSEXP, SEXP srcSEXP, SEXP dimSEXP, SEXP startSEXP, SEXP endSEXP, SEXP stepSEXP);
extern "C" SEXP _tinytorch_C_torch_select_scatter(SEXP selfSEXP, SEXP srcSEXP, SEXP dimSEXP, SEXP indexSEXP);
extern "C" SEXP _tinytorch_C_torch_diagonal_scatter(SEXP selfSEXP, SEXP srcSEXP, SEXP offsetSEXP, SEXP dim1SEXP, SEXP dim2SEXP);
extern "C" SEXP _tinytorch_C_torch_as_strided_scatter(SEXP selfSEXP, SEXP srcSEXP, SEXP size_sexpSEXP, SEXP stride_sexpSEXP, SEXP storage_offsetSEXP);
extern "C" SEXP _tinytorch_C_torch_smm(SEXP selfSEXP, SEXP mat2SEXP);
extern "C" SEXP _tinytorch_C_torch_unsafe_split(SEXP selfSEXP, SEXP split_sizeSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_split(SEXP selfSEXP, SEXP split_sizeSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_unsafe_split_with_sizes(SEXP selfSEXP, SEXP split_sizes_sexpSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_split_with_sizes(SEXP selfSEXP, SEXP split_sizes_sexpSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_hsplit(SEXP selfSEXP, SEXP sectionsSEXP);
extern "C" SEXP _tinytorch_C_torch_vsplit(SEXP selfSEXP, SEXP sectionsSEXP);
extern "C" SEXP _tinytorch_C_torch_dsplit(SEXP selfSEXP, SEXP sectionsSEXP);
extern "C" SEXP _tinytorch_C_torch_squeeze_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sspaddmm(SEXP selfSEXP, SEXP mat1SEXP, SEXP mat2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_stack(SEXP tensors_sexpSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_hstack(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_vstack(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_dstack(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_stride(SEXP selfSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_nansum(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_sum_to_size(SEXP selfSEXP, SEXP size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sqrt_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_square(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_square_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_std_mean(SEXP selfSEXP, SEXP unbiasedSEXP);
extern "C" SEXP _tinytorch_C_torch_prod(SEXP selfSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_t_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_tan(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_tan_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_tanh_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_tensordot(SEXP selfSEXP, SEXP otherSEXP, SEXP dims_self_sexpSEXP, SEXP dims_other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_threshold(SEXP selfSEXP, SEXP threshold_sexpSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_threshold_(SEXP selfSEXP, SEXP threshold_sexpSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_tile(SEXP selfSEXP, SEXP dims_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_transpose_(SEXP selfSEXP, SEXP dim0SEXP, SEXP dim1SEXP);
extern "C" SEXP _tinytorch_C_torch_one_hot(SEXP selfSEXP, SEXP num_classesSEXP);
extern "C" SEXP _tinytorch_C_torch_fliplr(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_flipud(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_roll(SEXP selfSEXP, SEXP shifts_sexpSEXP, SEXP dims_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_rot90(SEXP selfSEXP, SEXP kSEXP, SEXP dims_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_trapezoid(SEXP ySEXP, SEXP xSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_trapz(SEXP ySEXP, SEXP xSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_triplet_margin_loss(SEXP anchorSEXP, SEXP positiveSEXP, SEXP negativeSEXP, SEXP marginSEXP, SEXP pSEXP, SEXP epsSEXP, SEXP swapSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_trunc_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_fix(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_fix_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_type_as(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_unique_dim(SEXP selfSEXP, SEXP dimSEXP, SEXP sortedSEXP, SEXP return_inverseSEXP, SEXP return_countsSEXP);
extern "C" SEXP _tinytorch_C_torch_unique_consecutive(SEXP selfSEXP, SEXP return_inverseSEXP, SEXP return_countsSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_unique_dim_consecutive(SEXP selfSEXP, SEXP dimSEXP, SEXP return_inverseSEXP, SEXP return_countsSEXP);
extern "C" SEXP _tinytorch_C_torch_unsqueeze_(SEXP selfSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_vander(SEXP xSEXP, SEXP NSEXP, SEXP increasingSEXP);
extern "C" SEXP _tinytorch_C_torch_var(SEXP selfSEXP, SEXP unbiasedSEXP);
extern "C" SEXP _tinytorch_C_torch_var_mean(SEXP selfSEXP, SEXP unbiasedSEXP);
extern "C" SEXP _tinytorch_C_torch_view_as(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_norm_except_dim(SEXP vSEXP, SEXP powSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_poisson(SEXP selfSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_binomial(SEXP countSEXP, SEXP probSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_native_norm(SEXP selfSEXP, SEXP p_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_frexp(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_frobenius_norm(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_nuclear_norm(SEXP selfSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_positive(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_resize_as_(SEXP selfSEXP, SEXP the_templateSEXP, SEXP memory_formatSEXP);
extern "C" SEXP _tinytorch_C_torch_resize_as_sparse_(SEXP selfSEXP, SEXP the_templateSEXP);
extern "C" SEXP _tinytorch_C_torch_sub_(SEXP selfSEXP, SEXP otherSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_subtract(SEXP selfSEXP, SEXP otherSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_subtract_(SEXP selfSEXP, SEXP otherSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_rsub(SEXP selfSEXP, SEXP otherSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_heaviside(SEXP selfSEXP, SEXP valuesSEXP);
extern "C" SEXP _tinytorch_C_torch_heaviside_(SEXP selfSEXP, SEXP valuesSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_sampled_addmm(SEXP selfSEXP, SEXP mat1SEXP, SEXP mat2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addmm(SEXP selfSEXP, SEXP mat1SEXP, SEXP mat2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addmm_(SEXP selfSEXP, SEXP mat1SEXP, SEXP mat2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_compressed_tensor(SEXP compressed_indicesSEXP, SEXP plain_indicesSEXP, SEXP valuesSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_csr_tensor(SEXP crow_indicesSEXP, SEXP col_indicesSEXP, SEXP valuesSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_csc_tensor(SEXP ccol_indicesSEXP, SEXP row_indicesSEXP, SEXP valuesSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_bsr_tensor(SEXP crow_indicesSEXP, SEXP col_indicesSEXP, SEXP valuesSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_bsc_tensor(SEXP ccol_indicesSEXP, SEXP row_indicesSEXP, SEXP valuesSEXP, SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_coo_tensor(SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_resize_(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP sparse_dimSEXP, SEXP dense_dimSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_resize_and_clear_(SEXP selfSEXP, SEXP size_sexpSEXP, SEXP sparse_dimSEXP, SEXP dense_dimSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_mask(SEXP selfSEXP, SEXP maskSEXP);
extern "C" SEXP _tinytorch_C_torch_to_dense(SEXP selfSEXP, SEXP dtypeSEXP, SEXP masked_gradSEXP);
extern "C" SEXP _tinytorch_C_torch_sparse_dim(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_dense_dim(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_coalesce(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_is_coalesced(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_indices(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_values(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_crow_indices(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_col_indices(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_ccol_indices(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_row_indices(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_hspmm(SEXP mat1SEXP, SEXP mat2SEXP);
extern "C" SEXP _tinytorch_C_torch_copy_sparse_to_sparse_(SEXP selfSEXP, SEXP srcSEXP, SEXP non_blockingSEXP);
extern "C" SEXP _tinytorch_C_torch_unbind(SEXP selfSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_to_sparse(SEXP selfSEXP, SEXP sparse_dimSEXP);
extern "C" SEXP _tinytorch_C_torch_to_sparse_csr(SEXP selfSEXP, SEXP dense_dimSEXP);
extern "C" SEXP _tinytorch_C_torch_to_sparse_csc(SEXP selfSEXP, SEXP dense_dimSEXP);
extern "C" SEXP _tinytorch_C_torch_to_sparse_bsr(SEXP selfSEXP, SEXP blocksize_sexpSEXP, SEXP dense_dimSEXP);
extern "C" SEXP _tinytorch_C_torch_to_sparse_bsc(SEXP selfSEXP, SEXP blocksize_sexpSEXP, SEXP dense_dimSEXP);
extern "C" SEXP _tinytorch_C_torch_quantize_per_tensor_dynamic(SEXP selfSEXP, SEXP dtypeSEXP, SEXP reduce_rangeSEXP);
extern "C" SEXP _tinytorch_C_torch_quantize_per_tensor(SEXP selfSEXP, SEXP scaleSEXP, SEXP zero_pointSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_quantize_per_channel(SEXP selfSEXP, SEXP scalesSEXP, SEXP zero_pointsSEXP, SEXP axisSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_dequantize(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_q_scale(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_q_zero_point(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_q_per_channel_scales(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_q_per_channel_zero_points(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_q_per_channel_axis(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_int_repr(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_qscheme(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_fake_quantize_per_tensor_affine(SEXP selfSEXP, SEXP scaleSEXP, SEXP zero_pointSEXP, SEXP quant_minSEXP, SEXP quant_maxSEXP);
extern "C" SEXP _tinytorch_C_torch_fake_quantize_per_tensor_affine_cachemask(SEXP selfSEXP, SEXP scaleSEXP, SEXP zero_pointSEXP, SEXP quant_minSEXP, SEXP quant_maxSEXP);
extern "C" SEXP _tinytorch_C_torch_fake_quantize_per_channel_affine(SEXP selfSEXP, SEXP scaleSEXP, SEXP zero_pointSEXP, SEXP axisSEXP, SEXP quant_minSEXP, SEXP quant_maxSEXP);
extern "C" SEXP _tinytorch_C_torch_fake_quantize_per_channel_affine_cachemask(SEXP selfSEXP, SEXP scaleSEXP, SEXP zero_pointSEXP, SEXP axisSEXP, SEXP quant_minSEXP, SEXP quant_maxSEXP);
extern "C" SEXP _tinytorch_C_torch_fused_moving_avg_obs_fake_quant(SEXP selfSEXP, SEXP observer_onSEXP, SEXP fake_quant_onSEXP, SEXP running_minSEXP, SEXP running_maxSEXP, SEXP scaleSEXP, SEXP zero_pointSEXP, SEXP averaging_constSEXP, SEXP quant_minSEXP, SEXP quant_maxSEXP, SEXP ch_axisSEXP, SEXP per_row_fake_quantSEXP, SEXP symmetric_quantSEXP);
extern "C" SEXP _tinytorch_C_torch_choose_qparams_optimized(SEXP inputSEXP, SEXP numelSEXP, SEXP n_binsSEXP, SEXP ratioSEXP, SEXP bit_widthSEXP);
extern "C" SEXP _tinytorch_C_torch_meshgrid(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_cartesian_prod(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_combinations(SEXP selfSEXP, SEXP rSEXP, SEXP with_replacementSEXP);
extern "C" SEXP _tinytorch_C_torch_result_type(SEXP tensorSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_can_cast(SEXP from_SEXP, SEXP toSEXP);
extern "C" SEXP _tinytorch_C_torch_promote_types(SEXP type1SEXP, SEXP type2SEXP);
extern "C" SEXP _tinytorch_C_torch_gru(SEXP inputSEXP, SEXP hxSEXP, SEXP params_sexpSEXP, SEXP has_biasesSEXP, SEXP num_layersSEXP, SEXP dropoutSEXP, SEXP trainSEXP, SEXP bidirectionalSEXP, SEXP batch_firstSEXP);
extern "C" SEXP _tinytorch_C_torch_rnn_tanh(SEXP inputSEXP, SEXP hxSEXP, SEXP params_sexpSEXP, SEXP has_biasesSEXP, SEXP num_layersSEXP, SEXP dropoutSEXP, SEXP trainSEXP, SEXP bidirectionalSEXP, SEXP batch_firstSEXP);
extern "C" SEXP _tinytorch_C_torch_rnn_relu(SEXP inputSEXP, SEXP hxSEXP, SEXP params_sexpSEXP, SEXP has_biasesSEXP, SEXP num_layersSEXP, SEXP dropoutSEXP, SEXP trainSEXP, SEXP bidirectionalSEXP, SEXP batch_firstSEXP);
extern "C" SEXP _tinytorch_C_torch_lstm_cell(SEXP inputSEXP, SEXP hx_sexpSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP);
extern "C" SEXP _tinytorch_C_torch_gru_cell(SEXP inputSEXP, SEXP hxSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP);
extern "C" SEXP _tinytorch_C_torch_rnn_tanh_cell(SEXP inputSEXP, SEXP hxSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP);
extern "C" SEXP _tinytorch_C_torch_rnn_relu_cell(SEXP inputSEXP, SEXP hxSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_lstm_cell(SEXP inputSEXP, SEXP hx_sexpSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP, SEXP packed_ihSEXP, SEXP packed_hhSEXP, SEXP col_offsets_ihSEXP, SEXP col_offsets_hhSEXP, SEXP scale_ih_sexpSEXP, SEXP scale_hh_sexpSEXP, SEXP zero_point_ih_sexpSEXP, SEXP zero_point_hh_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_gru_cell(SEXP inputSEXP, SEXP hxSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP, SEXP packed_ihSEXP, SEXP packed_hhSEXP, SEXP col_offsets_ihSEXP, SEXP col_offsets_hhSEXP, SEXP scale_ih_sexpSEXP, SEXP scale_hh_sexpSEXP, SEXP zero_point_ih_sexpSEXP, SEXP zero_point_hh_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_rnn_relu_cell(SEXP inputSEXP, SEXP hxSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP, SEXP packed_ihSEXP, SEXP packed_hhSEXP, SEXP col_offsets_ihSEXP, SEXP col_offsets_hhSEXP, SEXP scale_ih_sexpSEXP, SEXP scale_hh_sexpSEXP, SEXP zero_point_ih_sexpSEXP, SEXP zero_point_hh_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_quantized_rnn_tanh_cell(SEXP inputSEXP, SEXP hxSEXP, SEXP w_ihSEXP, SEXP w_hhSEXP, SEXP b_ihSEXP, SEXP b_hhSEXP, SEXP packed_ihSEXP, SEXP packed_hhSEXP, SEXP col_offsets_ihSEXP, SEXP col_offsets_hhSEXP, SEXP scale_ih_sexpSEXP, SEXP scale_hh_sexpSEXP, SEXP zero_point_ih_sexpSEXP, SEXP zero_point_hh_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_set_(SEXP selfSEXP, SEXP sourceSEXP, SEXP storage_offsetSEXP, SEXP size_sexpSEXP, SEXP stride_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_is_set_to(SEXP selfSEXP, SEXP tensorSEXP);
extern "C" SEXP _tinytorch_C_torch_masked_scatter_(SEXP selfSEXP, SEXP maskSEXP, SEXP sourceSEXP);
extern "C" SEXP _tinytorch_C_torch_masked_scatter(SEXP selfSEXP, SEXP maskSEXP, SEXP sourceSEXP);
extern "C" SEXP _tinytorch_C_torch_put_(SEXP selfSEXP, SEXP indexSEXP, SEXP sourceSEXP, SEXP accumulateSEXP);
extern "C" SEXP _tinytorch_C_torch_put(SEXP selfSEXP, SEXP indexSEXP, SEXP sourceSEXP, SEXP accumulateSEXP);
extern "C" SEXP _tinytorch_C_torch_index_add_(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP sourceSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_index_add(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP sourceSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_index_reduce_(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP sourceSEXP, SEXP reduceSEXP, SEXP include_selfSEXP);
extern "C" SEXP _tinytorch_C_torch_index_reduce(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP sourceSEXP, SEXP reduceSEXP, SEXP include_selfSEXP);
extern "C" SEXP _tinytorch_C_torch_index_fill_(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_index_fill(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_scatter(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP srcSEXP);
extern "C" SEXP _tinytorch_C_torch_scatter_add(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP srcSEXP);
extern "C" SEXP _tinytorch_C_torch_scatter_add_(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP srcSEXP);
extern "C" SEXP _tinytorch_C_torch_scatter_reduce(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP srcSEXP, SEXP reduceSEXP, SEXP include_selfSEXP);
extern "C" SEXP _tinytorch_C_torch_scatter_reduce_(SEXP selfSEXP, SEXP dimSEXP, SEXP indexSEXP, SEXP srcSEXP, SEXP reduceSEXP, SEXP include_selfSEXP);
extern "C" SEXP _tinytorch_C_torch_eq_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_and(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_and_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___and__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___iand__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_or(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_or_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___or__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___ior__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_xor(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_xor_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___xor__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___ixor__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___lshift__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___ilshift__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_left_shift(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_left_shift_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch___rshift__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch___irshift__(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_right_shift(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_bitwise_right_shift_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_tril_(SEXP selfSEXP, SEXP diagonalSEXP);
extern "C" SEXP _tinytorch_C_torch_triu_(SEXP selfSEXP, SEXP diagonalSEXP);
extern "C" SEXP _tinytorch_C_torch_digamma_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_lerp_(SEXP selfSEXP, SEXP endSEXP, SEXP weight_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addbmm_(SEXP selfSEXP, SEXP batch1SEXP, SEXP batch2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addbmm(SEXP selfSEXP, SEXP batch1SEXP, SEXP batch2SEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_random_(SEXP selfSEXP, SEXP fromSEXP, SEXP toSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_cauchy_(SEXP selfSEXP, SEXP medianSEXP, SEXP sigmaSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_log_normal_(SEXP selfSEXP, SEXP meanSEXP, SEXP stdSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_exponential_(SEXP selfSEXP, SEXP lambdSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_geometric_(SEXP selfSEXP, SEXP pSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_diag(SEXP selfSEXP, SEXP diagonalSEXP);
extern "C" SEXP _tinytorch_C_torch_cross(SEXP selfSEXP, SEXP otherSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_tril(SEXP selfSEXP, SEXP diagonalSEXP);
extern "C" SEXP _tinytorch_C_torch_tril_indices(SEXP rowSEXP, SEXP colSEXP, SEXP offsetSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_triu_indices(SEXP rowSEXP, SEXP colSEXP, SEXP offsetSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_trace(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_ne_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_not_equal(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_not_equal_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_ge_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_greater_equal(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_greater_equal_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_le_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_less_equal(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_less_equal_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_gt_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_greater(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_greater_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_lt_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_less(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_less_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_take(SEXP selfSEXP, SEXP indexSEXP);
extern "C" SEXP _tinytorch_C_torch_take_along_dim(SEXP selfSEXP, SEXP indicesSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_masked_select(SEXP selfSEXP, SEXP maskSEXP);
extern "C" SEXP _tinytorch_C_torch_nonzero(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_nonzero_static(SEXP selfSEXP, SEXP sizeSEXP, SEXP fill_valueSEXP);
extern "C" SEXP _tinytorch_C_torch_nonzero_numpy(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_argwhere(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_addcmul(SEXP selfSEXP, SEXP tensor1SEXP, SEXP tensor2SEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addcmul_(SEXP selfSEXP, SEXP tensor1SEXP, SEXP tensor2SEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addcdiv(SEXP selfSEXP, SEXP tensor1SEXP, SEXP tensor2SEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addcdiv_(SEXP selfSEXP, SEXP tensor1SEXP, SEXP tensor2SEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_cross_entropy_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP reductionSEXP, SEXP ignore_indexSEXP, SEXP label_smoothingSEXP);
extern "C" SEXP _tinytorch_C_torch_triangular_solve(SEXP selfSEXP, SEXP ASEXP, SEXP upperSEXP, SEXP transposeSEXP, SEXP unitriangularSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_solve_triangular(SEXP selfSEXP, SEXP BSEXP, SEXP upperSEXP, SEXP leftSEXP, SEXP unitriangularSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_vander(SEXP xSEXP, SEXP NSEXP);
extern "C" SEXP _tinytorch_C_torch_svd(SEXP selfSEXP, SEXP someSEXP, SEXP compute_uvSEXP);
extern "C" SEXP _tinytorch_C_torch_swapaxes(SEXP selfSEXP, SEXP axis0SEXP, SEXP axis1SEXP);
extern "C" SEXP _tinytorch_C_torch_swapaxes_(SEXP selfSEXP, SEXP axis0SEXP, SEXP axis1SEXP);
extern "C" SEXP _tinytorch_C_torch_swapdims(SEXP selfSEXP, SEXP dim0SEXP, SEXP dim1SEXP);
extern "C" SEXP _tinytorch_C_torch_swapdims_(SEXP selfSEXP, SEXP dim0SEXP, SEXP dim1SEXP);
extern "C" SEXP _tinytorch_C_torch_cholesky(SEXP selfSEXP, SEXP upperSEXP);
extern "C" SEXP _tinytorch_C_torch_cholesky_solve(SEXP selfSEXP, SEXP input2SEXP, SEXP upperSEXP);
extern "C" SEXP _tinytorch_C_torch_cholesky_inverse(SEXP selfSEXP, SEXP upperSEXP);
extern "C" SEXP _tinytorch_C_torch_qr(SEXP selfSEXP, SEXP someSEXP);
extern "C" SEXP _tinytorch_C_torch_geqrf(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_orgqr(SEXP selfSEXP, SEXP input2SEXP);
extern "C" SEXP _tinytorch_C_torch_ormqr(SEXP selfSEXP, SEXP input2SEXP, SEXP input3SEXP, SEXP leftSEXP, SEXP transposeSEXP);
extern "C" SEXP _tinytorch_C_torch_lu_solve(SEXP selfSEXP, SEXP LU_dataSEXP, SEXP LU_pivotsSEXP);
extern "C" SEXP _tinytorch_C_torch_lu_unpack(SEXP LU_dataSEXP, SEXP LU_pivotsSEXP, SEXP unpack_dataSEXP, SEXP unpack_pivotsSEXP);
extern "C" SEXP _tinytorch_C_torch_lgamma_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_lgamma(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_digamma(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_polygamma(SEXP nSEXP, SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_polygamma_(SEXP selfSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_erfinv(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_erfinv_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_i0(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_i0_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sign_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_signbit(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_dist(SEXP selfSEXP, SEXP otherSEXP, SEXP p_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_atan2_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_atan2(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_arctan2(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_arctan2_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_lerp(SEXP selfSEXP, SEXP endSEXP, SEXP weight_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_histc(SEXP selfSEXP, SEXP binsSEXP, SEXP min_sexpSEXP, SEXP max_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_histogram(SEXP selfSEXP, SEXP binsSEXP, SEXP weightSEXP, SEXP densitySEXP);
extern "C" SEXP _tinytorch_C_torch_histogramdd(SEXP selfSEXP, SEXP bins_sexpSEXP, SEXP range_sexpSEXP, SEXP weightSEXP, SEXP densitySEXP);
extern "C" SEXP _tinytorch_C_torch_fmod(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_fmod_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_hypot(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_hypot_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_igamma(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_igamma_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_igammac(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_igammac_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_nextafter(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_nextafter_(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_remainder_(SEXP selfSEXP, SEXP other_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_fmin(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_fmax(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_minimum(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_quantile(SEXP selfSEXP, SEXP qSEXP, SEXP dimSEXP, SEXP keepdimSEXP, SEXP interpolationSEXP);
extern "C" SEXP _tinytorch_C_torch_nanquantile(SEXP selfSEXP, SEXP qSEXP, SEXP dimSEXP, SEXP keepdimSEXP, SEXP interpolationSEXP);
extern "C" SEXP _tinytorch_C_torch_msort(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_argsort(SEXP selfSEXP, SEXP dimSEXP, SEXP descendingSEXP);
extern "C" SEXP _tinytorch_C_torch_topk(SEXP selfSEXP, SEXP kSEXP, SEXP dimSEXP, SEXP largestSEXP, SEXP sortedSEXP);
extern "C" SEXP _tinytorch_C_torch_renorm(SEXP selfSEXP, SEXP p_sexpSEXP, SEXP dimSEXP, SEXP maxnorm_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_renorm_(SEXP selfSEXP, SEXP p_sexpSEXP, SEXP dimSEXP, SEXP maxnorm_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_unfold(SEXP selfSEXP, SEXP dimensionSEXP, SEXP sizeSEXP, SEXP stepSEXP);
extern "C" SEXP _tinytorch_C_torch_equal(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_pow_(SEXP selfSEXP, SEXP exponent_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_float_power(SEXP selfSEXP, SEXP exponentSEXP);
extern "C" SEXP _tinytorch_C_torch_float_power_(SEXP selfSEXP, SEXP exponent_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_normal_functional(SEXP selfSEXP, SEXP meanSEXP, SEXP stdSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_normal(SEXP meanSEXP, SEXP stdSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_alias(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_bucketize(SEXP selfSEXP, SEXP boundariesSEXP, SEXP out_int32SEXP, SEXP rightSEXP);
extern "C" SEXP _tinytorch_C_torch_searchsorted(SEXP sorted_sequenceSEXP, SEXP selfSEXP, SEXP out_int32SEXP, SEXP rightSEXP, SEXP sideSEXP, SEXP sorterSEXP);
extern "C" SEXP _tinytorch_C_torch_mse_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_l1_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_multi_margin_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP p_sexpSEXP, SEXP margin_sexpSEXP, SEXP weightSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_multilabel_margin_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_multilabel_margin_loss_forward(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_nll_loss_nd(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP reductionSEXP, SEXP ignore_indexSEXP);
extern "C" SEXP _tinytorch_C_torch_nll_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP reductionSEXP, SEXP ignore_indexSEXP);
extern "C" SEXP _tinytorch_C_torch_nll_loss_forward(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP reductionSEXP, SEXP ignore_indexSEXP);
extern "C" SEXP _tinytorch_C_torch_nll_loss2d(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP reductionSEXP, SEXP ignore_indexSEXP);
extern "C" SEXP _tinytorch_C_torch_nll_loss2d_forward(SEXP selfSEXP, SEXP targetSEXP, SEXP weightSEXP, SEXP reductionSEXP, SEXP ignore_indexSEXP);
extern "C" SEXP _tinytorch_C_torch_smooth_l1_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP, SEXP betaSEXP);
extern "C" SEXP _tinytorch_C_torch_huber_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP, SEXP deltaSEXP);
extern "C" SEXP _tinytorch_C_torch_soft_margin_loss(SEXP selfSEXP, SEXP targetSEXP, SEXP reductionSEXP);
extern "C" SEXP _tinytorch_C_torch_elu(SEXP selfSEXP, SEXP alpha_sexpSEXP, SEXP scale_sexpSEXP, SEXP input_scale_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_elu_(SEXP selfSEXP, SEXP alpha_sexpSEXP, SEXP scale_sexpSEXP, SEXP input_scale_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_glu(SEXP selfSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_glu_jvp(SEXP gluSEXP, SEXP xSEXP, SEXP dxSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_hardsigmoid(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_hardsigmoid_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_hardtanh(SEXP selfSEXP, SEXP min_val_sexpSEXP, SEXP max_val_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_hardtanh_(SEXP selfSEXP, SEXP min_val_sexpSEXP, SEXP max_val_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_hardswish(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_hardswish_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_leaky_relu(SEXP selfSEXP, SEXP negative_slope_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_leaky_relu_(SEXP selfSEXP, SEXP negative_slope_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_log_sigmoid(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log_sigmoid_forward(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_rrelu_with_noise(SEXP selfSEXP, SEXP noiseSEXP, SEXP lower_sexpSEXP, SEXP upper_sexpSEXP, SEXP trainingSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_rrelu_with_noise_(SEXP selfSEXP, SEXP noiseSEXP, SEXP lower_sexpSEXP, SEXP upper_sexpSEXP, SEXP trainingSEXP, SEXP generatorSEXP);
extern "C" SEXP _tinytorch_C_torch_softplus(SEXP selfSEXP, SEXP beta_sexpSEXP, SEXP threshold_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_softshrink(SEXP selfSEXP, SEXP lambd_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_adaptive_avg_pool2d(SEXP selfSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_adaptive_avg_pool3d(SEXP selfSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_adaptive_max_pool2d(SEXP selfSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_adaptive_max_pool3d(SEXP selfSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_avg_pool2d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP ceil_modeSEXP, SEXP count_include_padSEXP, SEXP divisor_overrideSEXP);
extern "C" SEXP _tinytorch_C_torch_avg_pool3d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP ceil_modeSEXP, SEXP count_include_padSEXP, SEXP divisor_overrideSEXP);
extern "C" SEXP _tinytorch_C_torch_fractional_max_pool2d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP output_size_sexpSEXP, SEXP random_samplesSEXP);
extern "C" SEXP _tinytorch_C_torch_fractional_max_pool3d(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP output_size_sexpSEXP, SEXP random_samplesSEXP);
extern "C" SEXP _tinytorch_C_torch_max_pool2d_with_indices(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_max_pool3d_with_indices(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP ceil_modeSEXP);
extern "C" SEXP _tinytorch_C_torch_max_unpool2d(SEXP selfSEXP, SEXP indicesSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_max_unpool3d(SEXP selfSEXP, SEXP indicesSEXP, SEXP output_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_reflection_pad1d(SEXP selfSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_reflection_pad2d(SEXP selfSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_reflection_pad3d(SEXP selfSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_replication_pad1d(SEXP selfSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_replication_pad2d(SEXP selfSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_replication_pad3d(SEXP selfSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_pad(SEXP selfSEXP, SEXP pad_sexpSEXP, SEXP modeSEXP, SEXP valueSEXP);
extern "C" SEXP _tinytorch_C_torch_upsample_linear1d(SEXP inputSEXP, SEXP output_size_sexpSEXP, SEXP align_cornersSEXP, SEXP scale_factors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_upsample_bilinear2d(SEXP inputSEXP, SEXP output_size_sexpSEXP, SEXP align_cornersSEXP, SEXP scale_factors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_upsample_trilinear3d(SEXP inputSEXP, SEXP output_size_sexpSEXP, SEXP align_cornersSEXP, SEXP scale_factors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_upsample_bicubic2d(SEXP inputSEXP, SEXP output_size_sexpSEXP, SEXP align_cornersSEXP, SEXP scale_factors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_upsample_nearest1d(SEXP inputSEXP, SEXP output_size_sexpSEXP, SEXP scale_factors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_upsample_nearest2d(SEXP inputSEXP, SEXP output_size_sexpSEXP, SEXP scale_factors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_upsample_nearest3d(SEXP inputSEXP, SEXP output_size_sexpSEXP, SEXP scale_factors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_slow_conv_transpose2d(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP output_padding_sexpSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_slow_conv_transpose3d(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP output_padding_sexpSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_thnn_conv2d(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_conv_depthwise3d(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_slow_conv3d(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_slow_conv3d_forward(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_slow_conv_dilated2d(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_slow_conv_dilated3d(SEXP selfSEXP, SEXP weightSEXP, SEXP kernel_size_sexpSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_col2im(SEXP selfSEXP, SEXP output_size_sexpSEXP, SEXP kernel_size_sexpSEXP, SEXP dilation_sexpSEXP, SEXP padding_sexpSEXP, SEXP stride_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_column_stack(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_im2col(SEXP selfSEXP, SEXP kernel_size_sexpSEXP, SEXP dilation_sexpSEXP, SEXP padding_sexpSEXP, SEXP stride_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_isfinite(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_isinf(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_isposinf(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_isneginf(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_entr(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_ndtri(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_log_ndtr(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_expm1(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_exp2(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_psi(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_digamma(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_gammaln(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_erf(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_erfc(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_erfcx(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_erfinv(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_ndtr(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_xlog1py(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_special_xlogy(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_special_zeta(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_special_i0(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_i0e(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_i1(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_i1e(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_logit(SEXP selfSEXP, SEXP epsSEXP);
extern "C" SEXP _tinytorch_C_torch_special_polygamma(SEXP nSEXP, SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_logsumexp(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP);
extern "C" SEXP _tinytorch_C_torch_special_expit(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_sinc(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_round(SEXP selfSEXP, SEXP decimalsSEXP);
extern "C" SEXP _tinytorch_C_torch_special_log1p(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_log_softmax(SEXP selfSEXP, SEXP dimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_special_gammainc(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_special_gammaincc(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_special_multigammaln(SEXP selfSEXP, SEXP pSEXP);
extern "C" SEXP _tinytorch_C_torch_special_softmax(SEXP selfSEXP, SEXP dimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_fft(SEXP selfSEXP, SEXP nSEXP, SEXP dimSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_ifft(SEXP selfSEXP, SEXP nSEXP, SEXP dimSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_rfft(SEXP selfSEXP, SEXP nSEXP, SEXP dimSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_irfft(SEXP selfSEXP, SEXP nSEXP, SEXP dimSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_hfft(SEXP selfSEXP, SEXP nSEXP, SEXP dimSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_ihfft(SEXP selfSEXP, SEXP nSEXP, SEXP dimSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_fft2(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_ifft2(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_rfft2(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_irfft2(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_hfft2(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_ihfft2(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_fftn(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_ifftn(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_rfftn(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_irfftn(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_hfftn(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_ihfftn(SEXP selfSEXP, SEXP s_sexpSEXP, SEXP dim_sexpSEXP, SEXP normSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_fftfreq(SEXP nSEXP, SEXP dSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_rfftfreq(SEXP nSEXP, SEXP dSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_fftshift(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_fft_ifftshift(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_cholesky_ex(SEXP selfSEXP, SEXP upperSEXP, SEXP check_errorsSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_cholesky(SEXP selfSEXP, SEXP upperSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_cross(SEXP selfSEXP, SEXP otherSEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_lu_factor(SEXP ASEXP, SEXP pivotSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_lu_factor_ex(SEXP ASEXP, SEXP pivotSEXP, SEXP check_errorsSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_lu(SEXP ASEXP, SEXP pivotSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_lu_solve(SEXP LUSEXP, SEXP pivotsSEXP, SEXP BSEXP, SEXP leftSEXP, SEXP adjointSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_det(SEXP ASEXP);
extern "C" SEXP _tinytorch_C_torch_det(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_ldl_factor_ex(SEXP selfSEXP, SEXP hermitianSEXP, SEXP check_errorsSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_ldl_factor(SEXP selfSEXP, SEXP hermitianSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_ldl_solve(SEXP LDSEXP, SEXP pivotsSEXP, SEXP BSEXP, SEXP hermitianSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_lstsq(SEXP selfSEXP, SEXP bSEXP, SEXP rcondSEXP, SEXP driverSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_matmul(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_vecdot(SEXP xSEXP, SEXP ySEXP, SEXP dimSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_matrix_exp(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_slogdet(SEXP ASEXP);
extern "C" SEXP _tinytorch_C_torch_slogdet(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_logdet(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_eig(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_eigvals(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_eigh(SEXP selfSEXP, SEXP UPLOSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_eigvalsh(SEXP selfSEXP, SEXP UPLOSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_householder_product(SEXP inputSEXP, SEXP tauSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_inv_ex(SEXP ASEXP, SEXP check_errorsSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_inv(SEXP ASEXP);
extern "C" SEXP _tinytorch_C_torch_inverse(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_inner(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_ger(SEXP selfSEXP, SEXP vec2SEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_norm(SEXP selfSEXP, SEXP ordSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_vector_norm(SEXP selfSEXP, SEXP ord_sexpSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_matrix_norm(SEXP selfSEXP, SEXP ord_sexpSEXP, SEXP dim_sexpSEXP, SEXP keepdimSEXP, SEXP dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_svd(SEXP ASEXP, SEXP full_matricesSEXP, SEXP driverSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_svdvals(SEXP ASEXP, SEXP driverSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_cond(SEXP selfSEXP, SEXP pSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_pinv(SEXP selfSEXP, SEXP atolSEXP, SEXP rtolSEXP, SEXP hermitianSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_solve_ex(SEXP ASEXP, SEXP BSEXP, SEXP leftSEXP, SEXP check_errorsSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_solve(SEXP ASEXP, SEXP BSEXP, SEXP leftSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_tensorinv(SEXP selfSEXP, SEXP indSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_tensorsolve(SEXP selfSEXP, SEXP otherSEXP, SEXP dims_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_qr(SEXP ASEXP, SEXP modeSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_matrix_power(SEXP selfSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_matrix_rank(SEXP inputSEXP, SEXP atolSEXP, SEXP rtolSEXP, SEXP hermitianSEXP);
extern "C" SEXP _tinytorch_C_torch_linalg_multi_dot(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_nested_to_padded_tensor(SEXP selfSEXP, SEXP paddingSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_segment_reduce(SEXP dataSEXP, SEXP reduceSEXP, SEXP lengthsSEXP, SEXP indicesSEXP, SEXP offsetsSEXP, SEXP axisSEXP, SEXP unsafeSEXP, SEXP initialSEXP);
extern "C" SEXP _tinytorch_C_torch_pad_sequence(SEXP sequences_sexpSEXP, SEXP batch_firstSEXP, SEXP padding_valueSEXP, SEXP padding_sideSEXP);
extern "C" SEXP _tinytorch_C_torch_flatten_dense_tensors(SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_unflatten_dense_tensors(SEXP flatSEXP, SEXP tensors_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_to_padded_tensor(SEXP selfSEXP, SEXP paddingSEXP, SEXP output_size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_special_airy_ai(SEXP xSEXP);
extern "C" SEXP _tinytorch_C_torch_special_bessel_j0(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_bessel_j1(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_bessel_y0(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_bessel_y1(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_chebyshev_polynomial_t(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_chebyshev_polynomial_u(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_chebyshev_polynomial_v(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_chebyshev_polynomial_w(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_hermite_polynomial_h(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_hermite_polynomial_he(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_laguerre_polynomial_l(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_legendre_polynomial_p(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_modified_bessel_i0(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_modified_bessel_i1(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_modified_bessel_k0(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_modified_bessel_k1(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_special_scaled_modified_bessel_k0(SEXP xSEXP);
extern "C" SEXP _tinytorch_C_torch_special_scaled_modified_bessel_k1(SEXP xSEXP);
extern "C" SEXP _tinytorch_C_torch_special_shifted_chebyshev_polynomial_t(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_shifted_chebyshev_polynomial_u(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_shifted_chebyshev_polynomial_v(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_shifted_chebyshev_polynomial_w(SEXP xSEXP, SEXP nSEXP);
extern "C" SEXP _tinytorch_C_torch_special_spherical_bessel_j0(SEXP xSEXP);
extern "C" SEXP _tinytorch_C_gpu_launch(SEXP ptx_sexpSEXP, SEXP kernel_name_sexpSEXP, SEXP inputs_sexpSEXP, SEXP output_sexpSEXP, SEXP grid_sexpSEXP, SEXP block_sexpSEXP, SEXP shared_mem_sexpSEXP);
extern "C" SEXP _tinytorch_C_gpu_launch_reduction(SEXP ptx_sexpSEXP, SEXP kernel_name_sexpSEXP, SEXP input_sexpSEXP, SEXP output_sexpSEXP, SEXP n_elements_sexpSEXP, SEXP grid_sexpSEXP, SEXP block_sexpSEXP, SEXP shared_mem_sexpSEXP);
extern "C" SEXP _tinytorch_C_gpu_launch_generic(SEXP ptx_sexpSEXP, SEXP kernel_name_sexpSEXP, SEXP tensors_sexpSEXP, SEXP scalars_sexpSEXP, SEXP grid_sexpSEXP, SEXP block_sexpSEXP, SEXP shared_mem_sexpSEXP);
extern "C" SEXP _tinytorch_C_gpu_kernel_cache_clear();
extern "C" SEXP _tinytorch_C_gpu_kernel_cache_stats();
extern "C" SEXP _tinytorch_C_torch_index(SEXP self_sexpSEXP, SEXP indices_listSEXP, SEXP drop_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_index_put(SEXP self_sexpSEXP, SEXP indices_listSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_silu(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_nnf_gelu(SEXP selfSEXP, SEXP approximate_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_leaky_relu(SEXP selfSEXP, SEXP negative_slope_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_elu(SEXP selfSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_softmax(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_log_softmax(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_layer_norm(SEXP selfSEXP, SEXP normalized_shape_sexpSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP eps_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_linear(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP);
extern "C" SEXP _tinytorch_C_torch_conv1d(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP groups_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_embedding(SEXP weightSEXP, SEXP indicesSEXP);
extern "C" SEXP _tinytorch_C_torch_conv_transpose1d(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP output_padding_sexpSEXP, SEXP groups_sexpSEXP, SEXP dilation_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_conv2d(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP dilation_sexpSEXP, SEXP groups_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_batch_norm(SEXP inputSEXP, SEXP weightSEXP, SEXP biasSEXP, SEXP running_meanSEXP, SEXP running_varSEXP, SEXP training_sexpSEXP, SEXP momentum_sexpSEXP, SEXP eps_sexpSEXP, SEXP cudnn_enabled_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_lstm(SEXP inputSEXP, SEXP hx_sexpSEXP, SEXP params_sexpSEXP, SEXP has_biases_sexpSEXP, SEXP num_layers_sexpSEXP, SEXP dropout_sexpSEXP, SEXP batch_first_sexpSEXP, SEXP bidirectional_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_pad(SEXP inputSEXP, SEXP pad_sexpSEXP, SEXP mode_sexpSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_interpolate(SEXP inputSEXP, SEXP size_sexpSEXP, SEXP scale_factor_sexpSEXP, SEXP mode_sexpSEXP, SEXP align_corners_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_avg_pool1d(SEXP inputSEXP, SEXP kernel_size_sexpSEXP, SEXP stride_sexpSEXP, SEXP padding_sexpSEXP, SEXP ceil_mode_sexpSEXP, SEXP count_include_pad_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_softplus(SEXP inputSEXP, SEXP beta_sexpSEXP, SEXP threshold_sexpSEXP);
extern "C" SEXP _tinytorch_C_nnf_normalize(SEXP inputSEXP, SEXP p_sexpSEXP, SEXP dim_sexpSEXP, SEXP eps_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_sdpa(SEXP querySEXP, SEXP keySEXP, SEXP valueSEXP, SEXP attn_mask_sexpSEXP, SEXP dropout_sexpSEXP, SEXP is_causal_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_add(SEXP selfSEXP, SEXP otherSEXP, SEXP alphaSEXP);
extern "C" SEXP _tinytorch_C_torch_sub(SEXP selfSEXP, SEXP otherSEXP, SEXP alphaSEXP);
extern "C" SEXP _tinytorch_C_torch_mul(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_div(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_neg(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_logical_not(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_add_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_sub_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_mul_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_div_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_matmul(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_mm(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_mm_dtype(SEXP selfSEXP, SEXP otherSEXP, SEXP out_dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_t(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sum(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_mean(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_max(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_min(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_argmax(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_argmin(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_reshape(SEXP selfSEXP, SEXP shape_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_view(SEXP selfSEXP, SEXP shape_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_squeeze(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_unsqueeze(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_relu(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sigmoid(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_tanh(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_exp(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log2(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_log10(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sqrt(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_abs(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sign(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_floor(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_ceil(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_round(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_trunc(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_sin(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_cos(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_rsqrt(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_detach(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_pow(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_pow_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_scalar_pow(SEXP scalarSEXP, SEXP exponentSEXP);
extern "C" SEXP _tinytorch_C_torch_remainder(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_remainder_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_floor_divide(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_floor_divide_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_eq(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_eq_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_ne(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_ne_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_lt(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_lt_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_le(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_le_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_gt(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_gt_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_ge(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_ge_scalar(SEXP selfSEXP, SEXP scalarSEXP);
extern "C" SEXP _tinytorch_C_torch_bmm(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_bmm_dtype(SEXP selfSEXP, SEXP otherSEXP, SEXP out_dtypeSEXP);
extern "C" SEXP _tinytorch_C_torch_transpose(SEXP selfSEXP, SEXP dim0_sexpSEXP, SEXP dim1_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_flatten(SEXP selfSEXP, SEXP start_dim_sexpSEXP, SEXP end_dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_clone(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_contiguous(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_to_dtype(SEXP selfSEXP, SEXP dtype_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_item(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_cat(SEXP tensors_sexpSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_clamp(SEXP selfSEXP, SEXP min_sexpSEXP, SEXP max_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_where(SEXP conditionSEXP, SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_sort(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP descending_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_flip(SEXP selfSEXP, SEXP dims_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_cumsum(SEXP selfSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_maximum(SEXP selfSEXP, SEXP otherSEXP);
extern "C" SEXP _tinytorch_C_torch_multinomial(SEXP selfSEXP, SEXP num_samples_sexpSEXP, SEXP replacement_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_outer(SEXP selfSEXP, SEXP vec2SEXP);
extern "C" SEXP _tinytorch_C_torch_triu(SEXP selfSEXP, SEXP diagonal_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_norm(SEXP selfSEXP, SEXP p_sexpSEXP, SEXP dim_sexpSEXP, SEXP keepdim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_std(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP keepdim_sexpSEXP, SEXP correction_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_complex(SEXP realSEXP, SEXP imagSEXP);
extern "C" SEXP _tinytorch_C_torch_real(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_imag(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_polar(SEXP absSEXP, SEXP angleSEXP);
extern "C" SEXP _tinytorch_C_torch_view_as_real(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_stft(SEXP inputSEXP, SEXP n_fft_sexpSEXP, SEXP hop_sexpSEXP, SEXP win_length_sexpSEXP, SEXP windowSEXP, SEXP center_sexpSEXP, SEXP normalized_sexpSEXP, SEXP onesided_sexpSEXP, SEXP return_complex_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_istft(SEXP inputSEXP, SEXP n_fft_sexpSEXP, SEXP hop_sexpSEXP, SEXP win_length_sexpSEXP, SEXP windowSEXP, SEXP center_sexpSEXP, SEXP normalized_sexpSEXP, SEXP onesided_sexpSEXP, SEXP length_sexpSEXP, SEXP return_complex_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_hann_window(SEXP length_sexpSEXP, SEXP periodic_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_addmm_dtype(SEXP selfSEXP, SEXP mat1SEXP, SEXP mat2SEXP, SEXP out_dtypeSEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_baddbmm_dtype(SEXP selfSEXP, SEXP batch1SEXP, SEXP batch2SEXP, SEXP out_dtypeSEXP, SEXP beta_sexpSEXP, SEXP alpha_sexpSEXP);
extern "C" SEXP _tinytorch_C_optim_sgd(SEXP params_sexpSEXP, SEXP lrSEXP, SEXP momentumSEXP, SEXP dampeningSEXP, SEXP weight_decaySEXP, SEXP nesterovSEXP);
extern "C" SEXP _tinytorch_C_optim_adam(SEXP params_sexpSEXP, SEXP lrSEXP, SEXP beta1SEXP, SEXP beta2SEXP, SEXP epsSEXP, SEXP weight_decaySEXP, SEXP amsgradSEXP);
extern "C" SEXP _tinytorch_C_optim_adamw(SEXP params_sexpSEXP, SEXP lrSEXP, SEXP beta1SEXP, SEXP beta2SEXP, SEXP epsSEXP, SEXP weight_decaySEXP, SEXP amsgradSEXP);
extern "C" SEXP _tinytorch_C_optim_step(SEXP optim_sexpSEXP);
extern "C" SEXP _tinytorch_C_optim_zero_grad(SEXP optim_sexpSEXP, SEXP set_to_noneSEXP);
extern "C" SEXP _tinytorch_C_rtorch_ping();
extern "C" SEXP _tinytorch_C_torch_tensor(SEXP dataSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_tensor_raw(SEXP dataSEXP, SEXP dim_sexpSEXP, SEXP dtype_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_zeros(SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_ones(SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_randn(SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_empty_like(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_empty(SEXP size_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_tensor_from_buffer(SEXP raw_sexpSEXP, SEXP shape_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_arange(SEXP start_sexpSEXP, SEXP end_sexpSEXP, SEXP step_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_full(SEXP size_sexpSEXP, SEXP fill_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_linspace(SEXP start_sexpSEXP, SEXP end_sexpSEXP, SEXP steps_sexpSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_ones_like(SEXP selfSEXP, SEXP dtype_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_zeros_like(SEXP selfSEXP, SEXP dtype_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_randn_like(SEXP selfSEXP, SEXP dtype_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_permute(SEXP selfSEXP, SEXP dims_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_expand(SEXP selfSEXP, SEXP size_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_gather(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP indexSEXP);
extern "C" SEXP _tinytorch_C_torch_masked_fill(SEXP selfSEXP, SEXP maskSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_masked_fill_(SEXP selfSEXP, SEXP maskSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_copy_(SEXP selfSEXP, SEXP srcSEXP);
extern "C" SEXP _tinytorch_C_torch_normal_(SEXP selfSEXP, SEXP mean_sexpSEXP, SEXP std_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_uniform_(SEXP selfSEXP, SEXP from_sexpSEXP, SEXP to_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_zero_(SEXP selfSEXP);
extern "C" SEXP _tinytorch_C_torch_fill_(SEXP selfSEXP, SEXP value_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_repeat(SEXP selfSEXP, SEXP sizes_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_repeat_interleave(SEXP selfSEXP, SEXP repeats_sexpSEXP, SEXP dim_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_index_select(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP indexSEXP);
extern "C" SEXP _tinytorch_C_torch_narrow(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP start_sexpSEXP, SEXP length_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_scatter_(SEXP selfSEXP, SEXP dim_sexpSEXP, SEXP indexSEXP, SEXP srcSEXP);
extern "C" SEXP _tinytorch_C_tensor_to_device(SEXP selfSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_tensor_to_dtype_device(SEXP selfSEXP, SEXP dtype_sexpSEXP, SEXP device_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_set_num_threads(SEXP n_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_get_num_threads();
extern "C" SEXP _tinytorch_C_torch_set_num_interop_threads(SEXP n_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_get_num_interop_threads();
extern "C" SEXP _tinytorch_C_cuda_is_available();
extern "C" SEXP _tinytorch_C_cuda_device_count();
extern "C" SEXP _tinytorch_C_cuda_empty_cache();
extern "C" SEXP _tinytorch_C_cuda_synchronize();
extern "C" SEXP _tinytorch_C_cuda_mem_info();
extern "C" SEXP _tinytorch_C_cuda_memory_stats();
extern "C" SEXP _tinytorch_C_torch_manual_seed(SEXP seed_sexpSEXP);
extern "C" SEXP _tinytorch_C_torch_scaled_mm(SEXP selfSEXP, SEXP mat2SEXP, SEXP scale_aSEXP, SEXP scale_bSEXP, SEXP biasSEXP, SEXP scale_resultSEXP, SEXP out_dtypeSEXP, SEXP use_fast_accumSEXP);
extern "C" SEXP _tinytorch_C_transformer_decoder_layer_step(SEXP x_sexpSEXP, SEXP weights_sexpSEXP, SEXP self_cache_k_sexpSEXP, SEXP self_cache_v_sexpSEXP, SEXP cross_cache_k_sexpSEXP, SEXP cross_cache_v_sexpSEXP, SEXP n_head_sexpSEXP);
extern "C" SEXP _tinytorch_C_transformer_encoder_layer(SEXP x_sexpSEXP, SEXP weights_sexpSEXP, SEXP n_head_sexpSEXP);
extern "C" SEXP _tinytorch_C_prepare_cross_caches(SEXP encoder_output_sexpSEXP, SEXP cross_kv_weightsSEXP, SEXP n_head_sexpSEXP);
extern "C" SEXP _tinytorch_C_encoder_forward(SEXP mel_sexpSEXP, SEXP global_weightsSEXP, SEXP layer_weightsSEXP, SEXP n_head_sexpSEXP, SEXP n_ctx_sexpSEXP);
extern "C" SEXP _tinytorch_C_decoder_forward_step(SEXP token_ids_sexpSEXP, SEXP global_weightsSEXP, SEXP layer_weightsSEXP, SEXP self_cache_k_listSEXP, SEXP self_cache_v_listSEXP, SEXP cross_cache_k_listSEXP, SEXP cross_cache_v_listSEXP, SEXP n_head_sexpSEXP, SEXP offset_sexpSEXP);
extern "C" SEXP _tinytorch_C_greedy_decode(SEXP initial_tokens_sexpSEXP, SEXP global_weightsSEXP, SEXP layer_weightsSEXP, SEXP cross_cache_k_listSEXP, SEXP cross_cache_v_listSEXP, SEXP n_head_sexpSEXP, SEXP max_length_sexpSEXP, SEXP eot_token_sexpSEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_tinytorch_C_tensor_requires_grad_", (DL_FUNC) &_tinytorch_C_tensor_requires_grad_, 2},
    {"_tinytorch_C_tensor_grad", (DL_FUNC) &_tinytorch_C_tensor_grad, 1},
    {"_tinytorch_C_tensor_backward", (DL_FUNC) &_tinytorch_C_tensor_backward, 4},
    {"_tinytorch_C_autograd_set_grad_mode", (DL_FUNC) &_tinytorch_C_autograd_set_grad_mode, 1},
    {"_tinytorch_C_autograd_is_enabled", (DL_FUNC) &_tinytorch_C_autograd_is_enabled, 0},
    {"_tinytorch_C_autograd_grad", (DL_FUNC) &_tinytorch_C_autograd_grad, 6},
    {"_tinytorch_C_tensor_is_leaf", (DL_FUNC) &_tinytorch_C_tensor_is_leaf, 1},
    {"_tinytorch_C_tensor_retain_grad", (DL_FUNC) &_tinytorch_C_tensor_retain_grad, 1},
    {"_tinytorch_C_as_array", (DL_FUNC) &_tinytorch_C_as_array, 1},
    {"_tinytorch_C_tensor_shape", (DL_FUNC) &_tinytorch_C_tensor_shape, 1},
    {"_tinytorch_C_tensor_dtype", (DL_FUNC) &_tinytorch_C_tensor_dtype, 1},
    {"_tinytorch_C_tensor_device", (DL_FUNC) &_tinytorch_C_tensor_device, 1},
    {"_tinytorch_C_tensor_ndim", (DL_FUNC) &_tinytorch_C_tensor_ndim, 1},
    {"_tinytorch_C_tensor_numel", (DL_FUNC) &_tinytorch_C_tensor_numel, 1},
    {"_tinytorch_C_tensor_requires_grad", (DL_FUNC) &_tinytorch_C_tensor_requires_grad, 1},
    {"_tinytorch_C_tensor_print", (DL_FUNC) &_tinytorch_C_tensor_print, 1},
    {"_tinytorch_cpp_fused_relu", (DL_FUNC) &_tinytorch_cpp_fused_relu, 1},
    {"_tinytorch_cpp_fused_relu_sigmoid", (DL_FUNC) &_tinytorch_cpp_fused_relu_sigmoid, 1},
    {"_tinytorch_cpp_fused_relu_sigmoid_tanh", (DL_FUNC) &_tinytorch_cpp_fused_relu_sigmoid_tanh, 1},
    {"_tinytorch_cpp_fused_silu", (DL_FUNC) &_tinytorch_cpp_fused_silu, 1},
    {"_tinytorch_cpp_fused_gelu", (DL_FUNC) &_tinytorch_cpp_fused_gelu, 1},
    {"_tinytorch_cpp_fused_sincos", (DL_FUNC) &_tinytorch_cpp_fused_sincos, 1},
    {"_tinytorch_cpp_fused_softcap", (DL_FUNC) &_tinytorch_cpp_fused_softcap, 2},
    {"_tinytorch_cpp_fused_rmsnorm", (DL_FUNC) &_tinytorch_cpp_fused_rmsnorm, 3},
    {"_tinytorch_cpp_tensor_shapes_key", (DL_FUNC) &_tinytorch_cpp_tensor_shapes_key, 1},
    {"_tinytorch_C_torch_rename_", (DL_FUNC) &_tinytorch_C_torch_rename_, 2},
    {"_tinytorch_C_torch_rename", (DL_FUNC) &_tinytorch_C_torch_rename, 2},
    {"_tinytorch_C_torch_align_to", (DL_FUNC) &_tinytorch_C_torch_align_to, 2},
    {"_tinytorch_C_torch_align_as", (DL_FUNC) &_tinytorch_C_torch_align_as, 2},
    {"_tinytorch_C_torch_align_tensors", (DL_FUNC) &_tinytorch_C_torch_align_tensors, 1},
    {"_tinytorch_C_torch_sym_constrain_range", (DL_FUNC) &_tinytorch_C_torch_sym_constrain_range, 3},
    {"_tinytorch_C_torch_sym_constrain_range_for_size", (DL_FUNC) &_tinytorch_C_torch_sym_constrain_range_for_size, 3},
    {"_tinytorch_C_torch_refine_names", (DL_FUNC) &_tinytorch_C_torch_refine_names, 2},
    {"_tinytorch_C_torch_native_dropout", (DL_FUNC) &_tinytorch_C_torch_native_dropout, 3},
    {"_tinytorch_C_torch_dropout", (DL_FUNC) &_tinytorch_C_torch_dropout, 3},
    {"_tinytorch_C_torch_dropout_", (DL_FUNC) &_tinytorch_C_torch_dropout_, 3},
    {"_tinytorch_C_torch_feature_dropout", (DL_FUNC) &_tinytorch_C_torch_feature_dropout, 3},
    {"_tinytorch_C_torch_feature_dropout_", (DL_FUNC) &_tinytorch_C_torch_feature_dropout_, 3},
    {"_tinytorch_C_torch_alpha_dropout", (DL_FUNC) &_tinytorch_C_torch_alpha_dropout, 3},
    {"_tinytorch_C_torch_alpha_dropout_", (DL_FUNC) &_tinytorch_C_torch_alpha_dropout_, 3},
    {"_tinytorch_C_torch_feature_alpha_dropout", (DL_FUNC) &_tinytorch_C_torch_feature_alpha_dropout, 3},
    {"_tinytorch_C_torch_feature_alpha_dropout_", (DL_FUNC) &_tinytorch_C_torch_feature_alpha_dropout_, 3},
    {"_tinytorch_C_torch_abs_", (DL_FUNC) &_tinytorch_C_torch_abs_, 1},
    {"_tinytorch_C_torch_absolute", (DL_FUNC) &_tinytorch_C_torch_absolute, 1},
    {"_tinytorch_C_torch_absolute_", (DL_FUNC) &_tinytorch_C_torch_absolute_, 1},
    {"_tinytorch_C_torch_angle", (DL_FUNC) &_tinytorch_C_torch_angle, 1},
    {"_tinytorch_C_torch_view_as_complex", (DL_FUNC) &_tinytorch_C_torch_view_as_complex, 1},
    {"_tinytorch_C_torch_sgn", (DL_FUNC) &_tinytorch_C_torch_sgn, 1},
    {"_tinytorch_C_torch_sgn_", (DL_FUNC) &_tinytorch_C_torch_sgn_, 1},
    {"_tinytorch_C_torch_chalf", (DL_FUNC) &_tinytorch_C_torch_chalf, 2},
    {"_tinytorch_C_torch_conj_physical", (DL_FUNC) &_tinytorch_C_torch_conj_physical, 1},
    {"_tinytorch_C_torch_conj_physical_", (DL_FUNC) &_tinytorch_C_torch_conj_physical_, 1},
    {"_tinytorch_C_torch_resolve_conj", (DL_FUNC) &_tinytorch_C_torch_resolve_conj, 1},
    {"_tinytorch_C_torch_resolve_neg", (DL_FUNC) &_tinytorch_C_torch_resolve_neg, 1},
    {"_tinytorch_C_torch_acos", (DL_FUNC) &_tinytorch_C_torch_acos, 1},
    {"_tinytorch_C_torch_acos_", (DL_FUNC) &_tinytorch_C_torch_acos_, 1},
    {"_tinytorch_C_torch_arccos", (DL_FUNC) &_tinytorch_C_torch_arccos, 1},
    {"_tinytorch_C_torch_arccos_", (DL_FUNC) &_tinytorch_C_torch_arccos_, 1},
    {"_tinytorch_C_torch_avg_pool1d", (DL_FUNC) &_tinytorch_C_torch_avg_pool1d, 6},
    {"_tinytorch_C_torch_adaptive_avg_pool1d", (DL_FUNC) &_tinytorch_C_torch_adaptive_avg_pool1d, 2},
    {"_tinytorch_C_torch_adaptive_max_pool1d", (DL_FUNC) &_tinytorch_C_torch_adaptive_max_pool1d, 2},
    {"_tinytorch_C_torch_add_", (DL_FUNC) &_tinytorch_C_torch_add_, 3},
    {"_tinytorch_C_torch_addmv", (DL_FUNC) &_tinytorch_C_torch_addmv, 5},
    {"_tinytorch_C_torch_addmv_", (DL_FUNC) &_tinytorch_C_torch_addmv_, 5},
    {"_tinytorch_C_torch_addr", (DL_FUNC) &_tinytorch_C_torch_addr, 5},
    {"_tinytorch_C_torch_addr_", (DL_FUNC) &_tinytorch_C_torch_addr_, 5},
    {"_tinytorch_C_torch_affine_grid_generator", (DL_FUNC) &_tinytorch_C_torch_affine_grid_generator, 3},
    {"_tinytorch_C_torch_all", (DL_FUNC) &_tinytorch_C_torch_all, 3},
    {"_tinytorch_C_torch_any", (DL_FUNC) &_tinytorch_C_torch_any, 3},
    {"_tinytorch_C_torch_acosh", (DL_FUNC) &_tinytorch_C_torch_acosh, 1},
    {"_tinytorch_C_torch_acosh_", (DL_FUNC) &_tinytorch_C_torch_acosh_, 1},
    {"_tinytorch_C_torch_arccosh", (DL_FUNC) &_tinytorch_C_torch_arccosh, 1},
    {"_tinytorch_C_torch_arccosh_", (DL_FUNC) &_tinytorch_C_torch_arccosh_, 1},
    {"_tinytorch_C_torch_asinh", (DL_FUNC) &_tinytorch_C_torch_asinh, 1},
    {"_tinytorch_C_torch_asinh_", (DL_FUNC) &_tinytorch_C_torch_asinh_, 1},
    {"_tinytorch_C_torch_arcsinh", (DL_FUNC) &_tinytorch_C_torch_arcsinh, 1},
    {"_tinytorch_C_torch_arcsinh_", (DL_FUNC) &_tinytorch_C_torch_arcsinh_, 1},
    {"_tinytorch_C_torch_atanh", (DL_FUNC) &_tinytorch_C_torch_atanh, 1},
    {"_tinytorch_C_torch_atanh_", (DL_FUNC) &_tinytorch_C_torch_atanh_, 1},
    {"_tinytorch_C_torch_arctanh", (DL_FUNC) &_tinytorch_C_torch_arctanh, 1},
    {"_tinytorch_C_torch_arctanh_", (DL_FUNC) &_tinytorch_C_torch_arctanh_, 1},
    {"_tinytorch_C_torch_as_strided", (DL_FUNC) &_tinytorch_C_torch_as_strided, 4},
    {"_tinytorch_C_torch_as_strided_", (DL_FUNC) &_tinytorch_C_torch_as_strided_, 4},
    {"_tinytorch_C_torch_asin", (DL_FUNC) &_tinytorch_C_torch_asin, 1},
    {"_tinytorch_C_torch_asin_", (DL_FUNC) &_tinytorch_C_torch_asin_, 1},
    {"_tinytorch_C_torch_arcsin", (DL_FUNC) &_tinytorch_C_torch_arcsin, 1},
    {"_tinytorch_C_torch_arcsin_", (DL_FUNC) &_tinytorch_C_torch_arcsin_, 1},
    {"_tinytorch_C_torch_atan", (DL_FUNC) &_tinytorch_C_torch_atan, 1},
    {"_tinytorch_C_torch_atan_", (DL_FUNC) &_tinytorch_C_torch_atan_, 1},
    {"_tinytorch_C_torch_arctan", (DL_FUNC) &_tinytorch_C_torch_arctan, 1},
    {"_tinytorch_C_torch_arctan_", (DL_FUNC) &_tinytorch_C_torch_arctan_, 1},
    {"_tinytorch_C_torch_atleast_1d", (DL_FUNC) &_tinytorch_C_torch_atleast_1d, 1},
    {"_tinytorch_C_torch_atleast_2d", (DL_FUNC) &_tinytorch_C_torch_atleast_2d, 1},
    {"_tinytorch_C_torch_atleast_3d", (DL_FUNC) &_tinytorch_C_torch_atleast_3d, 1},
    {"_tinytorch_C_torch_baddbmm", (DL_FUNC) &_tinytorch_C_torch_baddbmm, 5},
    {"_tinytorch_C_torch_baddbmm_", (DL_FUNC) &_tinytorch_C_torch_baddbmm_, 5},
    {"_tinytorch_C_torch_bartlett_window", (DL_FUNC) &_tinytorch_C_torch_bartlett_window, 3},
    {"_tinytorch_C_torch_quantized_batch_norm", (DL_FUNC) &_tinytorch_C_torch_quantized_batch_norm, 8},
    {"_tinytorch_C_torch_bernoulli", (DL_FUNC) &_tinytorch_C_torch_bernoulli, 2},
    {"_tinytorch_C_torch_bernoulli_", (DL_FUNC) &_tinytorch_C_torch_bernoulli_, 3},
    {"_tinytorch_C_torch_bilinear", (DL_FUNC) &_tinytorch_C_torch_bilinear, 4},
    {"_tinytorch_C_torch_binary_cross_entropy", (DL_FUNC) &_tinytorch_C_torch_binary_cross_entropy, 4},
    {"_tinytorch_C_torch_binary_cross_entropy_with_logits", (DL_FUNC) &_tinytorch_C_torch_binary_cross_entropy_with_logits, 5},
    {"_tinytorch_C_torch_bincount", (DL_FUNC) &_tinytorch_C_torch_bincount, 3},
    {"_tinytorch_C_torch_bitwise_not", (DL_FUNC) &_tinytorch_C_torch_bitwise_not, 1},
    {"_tinytorch_C_torch_bitwise_not_", (DL_FUNC) &_tinytorch_C_torch_bitwise_not_, 1},
    {"_tinytorch_C_torch_copysign", (DL_FUNC) &_tinytorch_C_torch_copysign, 2},
    {"_tinytorch_C_torch_copysign_", (DL_FUNC) &_tinytorch_C_torch_copysign_, 2},
    {"_tinytorch_C_torch_logical_not_", (DL_FUNC) &_tinytorch_C_torch_logical_not_, 1},
    {"_tinytorch_C_torch_logical_xor", (DL_FUNC) &_tinytorch_C_torch_logical_xor, 2},
    {"_tinytorch_C_torch_logical_xor_", (DL_FUNC) &_tinytorch_C_torch_logical_xor_, 2},
    {"_tinytorch_C_torch_logical_and", (DL_FUNC) &_tinytorch_C_torch_logical_and, 2},
    {"_tinytorch_C_torch_logical_and_", (DL_FUNC) &_tinytorch_C_torch_logical_and_, 2},
    {"_tinytorch_C_torch_logical_or", (DL_FUNC) &_tinytorch_C_torch_logical_or, 2},
    {"_tinytorch_C_torch_logical_or_", (DL_FUNC) &_tinytorch_C_torch_logical_or_, 2},
    {"_tinytorch_C_torch_blackman_window", (DL_FUNC) &_tinytorch_C_torch_blackman_window, 3},
    {"_tinytorch_C_torch_broadcast_tensors", (DL_FUNC) &_tinytorch_C_torch_broadcast_tensors, 1},
    {"_tinytorch_C_torch_broadcast_to", (DL_FUNC) &_tinytorch_C_torch_broadcast_to, 2},
    {"_tinytorch_C_torch_concat", (DL_FUNC) &_tinytorch_C_torch_concat, 2},
    {"_tinytorch_C_torch_concatenate", (DL_FUNC) &_tinytorch_C_torch_concatenate, 2},
    {"_tinytorch_C_torch_block_diag", (DL_FUNC) &_tinytorch_C_torch_block_diag, 1},
    {"_tinytorch_C_torch_ceil_", (DL_FUNC) &_tinytorch_C_torch_ceil_, 1},
    {"_tinytorch_C_torch_chain_matmul", (DL_FUNC) &_tinytorch_C_torch_chain_matmul, 1},
    {"_tinytorch_C_torch_unsafe_chunk", (DL_FUNC) &_tinytorch_C_torch_unsafe_chunk, 3},
    {"_tinytorch_C_torch_chunk", (DL_FUNC) &_tinytorch_C_torch_chunk, 3},
    {"_tinytorch_C_torch_tensor_split", (DL_FUNC) &_tinytorch_C_torch_tensor_split, 3},
    {"_tinytorch_C_torch_clamp_", (DL_FUNC) &_tinytorch_C_torch_clamp_, 3},
    {"_tinytorch_C_torch_clamp_max", (DL_FUNC) &_tinytorch_C_torch_clamp_max, 2},
    {"_tinytorch_C_torch_clamp_max_", (DL_FUNC) &_tinytorch_C_torch_clamp_max_, 2},
    {"_tinytorch_C_torch_clamp_min", (DL_FUNC) &_tinytorch_C_torch_clamp_min, 2},
    {"_tinytorch_C_torch_clamp_min_", (DL_FUNC) &_tinytorch_C_torch_clamp_min_, 2},
    {"_tinytorch_C_torch_clip", (DL_FUNC) &_tinytorch_C_torch_clip, 3},
    {"_tinytorch_C_torch_clip_", (DL_FUNC) &_tinytorch_C_torch_clip_, 3},
    {"_tinytorch_C_torch_constant_pad_nd", (DL_FUNC) &_tinytorch_C_torch_constant_pad_nd, 3},
    {"_tinytorch_C_torch_convolution", (DL_FUNC) &_tinytorch_C_torch_convolution, 9},
    {"_tinytorch_C_torch_convolution_overrideable", (DL_FUNC) &_tinytorch_C_torch_convolution_overrideable, 9},
    {"_tinytorch_C_torch_conv3d", (DL_FUNC) &_tinytorch_C_torch_conv3d, 7},
    {"_tinytorch_C_torch_conv_tbc", (DL_FUNC) &_tinytorch_C_torch_conv_tbc, 4},
    {"_tinytorch_C_torch_conv_transpose2d", (DL_FUNC) &_tinytorch_C_torch_conv_transpose2d, 8},
    {"_tinytorch_C_torch_conv_transpose3d", (DL_FUNC) &_tinytorch_C_torch_conv_transpose3d, 8},
    {"_tinytorch_C_torch_copy", (DL_FUNC) &_tinytorch_C_torch_copy, 3},
    {"_tinytorch_C_torch_cos_", (DL_FUNC) &_tinytorch_C_torch_cos_, 1},
    {"_tinytorch_C_torch_cosh", (DL_FUNC) &_tinytorch_C_torch_cosh, 1},
    {"_tinytorch_C_torch_cosh_", (DL_FUNC) &_tinytorch_C_torch_cosh_, 1},
    {"_tinytorch_C_torch_cosine_embedding_loss", (DL_FUNC) &_tinytorch_C_torch_cosine_embedding_loss, 5},
    {"_tinytorch_C_torch_count_nonzero", (DL_FUNC) &_tinytorch_C_torch_count_nonzero, 2},
    {"_tinytorch_C_torch_cov", (DL_FUNC) &_tinytorch_C_torch_cov, 4},
    {"_tinytorch_C_torch_corrcoef", (DL_FUNC) &_tinytorch_C_torch_corrcoef, 1},
    {"_tinytorch_C_torch_cummax", (DL_FUNC) &_tinytorch_C_torch_cummax, 2},
    {"_tinytorch_C_torch_cummin", (DL_FUNC) &_tinytorch_C_torch_cummin, 2},
    {"_tinytorch_C_torch_cumprod", (DL_FUNC) &_tinytorch_C_torch_cumprod, 3},
    {"_tinytorch_C_torch_cumprod_", (DL_FUNC) &_tinytorch_C_torch_cumprod_, 3},
    {"_tinytorch_C_torch_cumsum_", (DL_FUNC) &_tinytorch_C_torch_cumsum_, 3},
    {"_tinytorch_C_torch_cumulative_trapezoid", (DL_FUNC) &_tinytorch_C_torch_cumulative_trapezoid, 3},
    {"_tinytorch_C_torch_ctc_loss", (DL_FUNC) &_tinytorch_C_torch_ctc_loss, 7},
    {"_tinytorch_C_torch_diag_embed", (DL_FUNC) &_tinytorch_C_torch_diag_embed, 4},
    {"_tinytorch_C_torch_diagflat", (DL_FUNC) &_tinytorch_C_torch_diagflat, 2},
    {"_tinytorch_C_torch_diagonal", (DL_FUNC) &_tinytorch_C_torch_diagonal, 4},
    {"_tinytorch_C_torch_linalg_diagonal", (DL_FUNC) &_tinytorch_C_torch_linalg_diagonal, 4},
    {"_tinytorch_C_torch_fill_diagonal_", (DL_FUNC) &_tinytorch_C_torch_fill_diagonal_, 3},
    {"_tinytorch_C_torch_diff", (DL_FUNC) &_tinytorch_C_torch_diff, 5},
    {"_tinytorch_C_torch_gradient", (DL_FUNC) &_tinytorch_C_torch_gradient, 4},
    {"_tinytorch_C_torch_div_", (DL_FUNC) &_tinytorch_C_torch_div_, 2},
    {"_tinytorch_C_torch_divide", (DL_FUNC) &_tinytorch_C_torch_divide, 2},
    {"_tinytorch_C_torch_divide_", (DL_FUNC) &_tinytorch_C_torch_divide_, 2},
    {"_tinytorch_C_torch_true_divide", (DL_FUNC) &_tinytorch_C_torch_true_divide, 2},
    {"_tinytorch_C_torch_true_divide_", (DL_FUNC) &_tinytorch_C_torch_true_divide_, 2},
    {"_tinytorch_C_torch_dot", (DL_FUNC) &_tinytorch_C_torch_dot, 2},
    {"_tinytorch_C_torch_vdot", (DL_FUNC) &_tinytorch_C_torch_vdot, 2},
    {"_tinytorch_C_torch_einsum", (DL_FUNC) &_tinytorch_C_torch_einsum, 3},
    {"_tinytorch_C_torch_embedding_renorm_", (DL_FUNC) &_tinytorch_C_torch_embedding_renorm_, 4},
    {"_tinytorch_C_torch_row_stack", (DL_FUNC) &_tinytorch_C_torch_row_stack, 1},
    {"_tinytorch_C_torch_embedding_bag", (DL_FUNC) &_tinytorch_C_torch_embedding_bag, 8},
    {"_tinytorch_C_torch_empty_permuted", (DL_FUNC) &_tinytorch_C_torch_empty_permuted, 4},
    {"_tinytorch_C_torch_new_empty", (DL_FUNC) &_tinytorch_C_torch_new_empty, 4},
    {"_tinytorch_C_torch_new_empty_strided", (DL_FUNC) &_tinytorch_C_torch_new_empty_strided, 5},
    {"_tinytorch_C_torch_new_full", (DL_FUNC) &_tinytorch_C_torch_new_full, 5},
    {"_tinytorch_C_torch_new_zeros", (DL_FUNC) &_tinytorch_C_torch_new_zeros, 4},
    {"_tinytorch_C_torch_new_ones", (DL_FUNC) &_tinytorch_C_torch_new_ones, 4},
    {"_tinytorch_C_torch_resize_", (DL_FUNC) &_tinytorch_C_torch_resize_, 3},
    {"_tinytorch_C_torch_empty_quantized", (DL_FUNC) &_tinytorch_C_torch_empty_quantized, 5},
    {"_tinytorch_C_torch_empty_strided", (DL_FUNC) &_tinytorch_C_torch_empty_strided, 4},
    {"_tinytorch_C_torch_erf", (DL_FUNC) &_tinytorch_C_torch_erf, 1},
    {"_tinytorch_C_torch_erf_", (DL_FUNC) &_tinytorch_C_torch_erf_, 1},
    {"_tinytorch_C_torch_erfc", (DL_FUNC) &_tinytorch_C_torch_erfc, 1},
    {"_tinytorch_C_torch_erfc_", (DL_FUNC) &_tinytorch_C_torch_erfc_, 1},
    {"_tinytorch_C_torch_exp_", (DL_FUNC) &_tinytorch_C_torch_exp_, 1},
    {"_tinytorch_C_torch_exp2", (DL_FUNC) &_tinytorch_C_torch_exp2, 1},
    {"_tinytorch_C_torch_exp2_", (DL_FUNC) &_tinytorch_C_torch_exp2_, 1},
    {"_tinytorch_C_torch_expm1", (DL_FUNC) &_tinytorch_C_torch_expm1, 1},
    {"_tinytorch_C_torch_expm1_", (DL_FUNC) &_tinytorch_C_torch_expm1_, 1},
    {"_tinytorch_C_torch_expand_as", (DL_FUNC) &_tinytorch_C_torch_expand_as, 2},
    {"_tinytorch_C_torch_eye", (DL_FUNC) &_tinytorch_C_torch_eye, 3},
    {"_tinytorch_C_torch_unflatten", (DL_FUNC) &_tinytorch_C_torch_unflatten, 3},
    {"_tinytorch_C_torch_fill", (DL_FUNC) &_tinytorch_C_torch_fill, 2},
    {"_tinytorch_C_torch_floor_", (DL_FUNC) &_tinytorch_C_torch_floor_, 1},
    {"_tinytorch_C_torch_floor_divide_", (DL_FUNC) &_tinytorch_C_torch_floor_divide_, 2},
    {"_tinytorch_C_torch_frac", (DL_FUNC) &_tinytorch_C_torch_frac, 1},
    {"_tinytorch_C_torch_frac_", (DL_FUNC) &_tinytorch_C_torch_frac_, 1},
    {"_tinytorch_C_torch_full_like", (DL_FUNC) &_tinytorch_C_torch_full_like, 5},
    {"_tinytorch_C_torch_from_file", (DL_FUNC) &_tinytorch_C_torch_from_file, 5},
    {"_tinytorch_C_torch_gcd", (DL_FUNC) &_tinytorch_C_torch_gcd, 2},
    {"_tinytorch_C_torch_gcd_", (DL_FUNC) &_tinytorch_C_torch_gcd_, 2},
    {"_tinytorch_C_torch_lcm", (DL_FUNC) &_tinytorch_C_torch_lcm, 2},
    {"_tinytorch_C_torch_lcm_", (DL_FUNC) &_tinytorch_C_torch_lcm_, 2},
    {"_tinytorch_C_torch_grid_sampler", (DL_FUNC) &_tinytorch_C_torch_grid_sampler, 5},
    {"_tinytorch_C_torch_grid_sampler_2d", (DL_FUNC) &_tinytorch_C_torch_grid_sampler_2d, 5},
    {"_tinytorch_C_torch_grid_sampler_3d", (DL_FUNC) &_tinytorch_C_torch_grid_sampler_3d, 5},
    {"_tinytorch_C_torch_hamming_window", (DL_FUNC) &_tinytorch_C_torch_hamming_window, 3},
    {"_tinytorch_C_torch_kaiser_window", (DL_FUNC) &_tinytorch_C_torch_kaiser_window, 3},
    {"_tinytorch_C_torch_hinge_embedding_loss", (DL_FUNC) &_tinytorch_C_torch_hinge_embedding_loss, 4},
    {"_tinytorch_C_torch_group_norm", (DL_FUNC) &_tinytorch_C_torch_group_norm, 6},
    {"_tinytorch_C_torch_native_group_norm", (DL_FUNC) &_tinytorch_C_torch_native_group_norm, 8},
    {"_tinytorch_C_torch_index_copy_", (DL_FUNC) &_tinytorch_C_torch_index_copy_, 4},
    {"_tinytorch_C_torch_index_put_", (DL_FUNC) &_tinytorch_C_torch_index_put_, 4},
    {"_tinytorch_C_torch_instance_norm", (DL_FUNC) &_tinytorch_C_torch_instance_norm, 9},
    {"_tinytorch_C_torch_isclose", (DL_FUNC) &_tinytorch_C_torch_isclose, 5},
    {"_tinytorch_C_torch_isin", (DL_FUNC) &_tinytorch_C_torch_isin, 4},
    {"_tinytorch_C_torch_isnan", (DL_FUNC) &_tinytorch_C_torch_isnan, 1},
    {"_tinytorch_C_torch_is_distributed", (DL_FUNC) &_tinytorch_C_torch_is_distributed, 1},
    {"_tinytorch_C_torch_isreal", (DL_FUNC) &_tinytorch_C_torch_isreal, 1},
    {"_tinytorch_C_torch_is_nonzero", (DL_FUNC) &_tinytorch_C_torch_is_nonzero, 1},
    {"_tinytorch_C_torch_is_same_size", (DL_FUNC) &_tinytorch_C_torch_is_same_size, 2},
    {"_tinytorch_C_torch_kl_div", (DL_FUNC) &_tinytorch_C_torch_kl_div, 4},
    {"_tinytorch_C_torch_kron", (DL_FUNC) &_tinytorch_C_torch_kron, 2},
    {"_tinytorch_C_torch_kthvalue", (DL_FUNC) &_tinytorch_C_torch_kthvalue, 4},
    {"_tinytorch_C_torch_native_layer_norm", (DL_FUNC) &_tinytorch_C_torch_native_layer_norm, 5},
    {"_tinytorch_C_torch_rms_norm", (DL_FUNC) &_tinytorch_C_torch_rms_norm, 4},
    {"_tinytorch_C_torch_nan_to_num", (DL_FUNC) &_tinytorch_C_torch_nan_to_num, 4},
    {"_tinytorch_C_torch_nan_to_num_", (DL_FUNC) &_tinytorch_C_torch_nan_to_num_, 4},
    {"_tinytorch_C_torch_ldexp", (DL_FUNC) &_tinytorch_C_torch_ldexp, 2},
    {"_tinytorch_C_torch_ldexp_", (DL_FUNC) &_tinytorch_C_torch_ldexp_, 2},
    {"_tinytorch_C_torch_log_", (DL_FUNC) &_tinytorch_C_torch_log_, 1},
    {"_tinytorch_C_torch_log10_", (DL_FUNC) &_tinytorch_C_torch_log10_, 1},
    {"_tinytorch_C_torch_log1p", (DL_FUNC) &_tinytorch_C_torch_log1p, 1},
    {"_tinytorch_C_torch_log1p_", (DL_FUNC) &_tinytorch_C_torch_log1p_, 1},
    {"_tinytorch_C_torch_log2_", (DL_FUNC) &_tinytorch_C_torch_log2_, 1},
    {"_tinytorch_C_torch_logaddexp", (DL_FUNC) &_tinytorch_C_torch_logaddexp, 2},
    {"_tinytorch_C_torch_logaddexp2", (DL_FUNC) &_tinytorch_C_torch_logaddexp2, 2},
    {"_tinytorch_C_torch_xlogy", (DL_FUNC) &_tinytorch_C_torch_xlogy, 2},
    {"_tinytorch_C_torch_xlogy_", (DL_FUNC) &_tinytorch_C_torch_xlogy_, 2},
    {"_tinytorch_C_torch_logspace", (DL_FUNC) &_tinytorch_C_torch_logspace, 6},
    {"_tinytorch_C_torch_logcumsumexp", (DL_FUNC) &_tinytorch_C_torch_logcumsumexp, 2},
    {"_tinytorch_C_torch_logsumexp", (DL_FUNC) &_tinytorch_C_torch_logsumexp, 3},
    {"_tinytorch_C_torch_margin_ranking_loss", (DL_FUNC) &_tinytorch_C_torch_margin_ranking_loss, 5},
    {"_tinytorch_C_torch_matrix_power", (DL_FUNC) &_tinytorch_C_torch_matrix_power, 2},
    {"_tinytorch_C_torch_matrix_exp", (DL_FUNC) &_tinytorch_C_torch_matrix_exp, 1},
    {"_tinytorch_C_torch_aminmax", (DL_FUNC) &_tinytorch_C_torch_aminmax, 3},
    {"_tinytorch_C_torch_amax", (DL_FUNC) &_tinytorch_C_torch_amax, 3},
    {"_tinytorch_C_torch_max_pool1d_with_indices", (DL_FUNC) &_tinytorch_C_torch_max_pool1d_with_indices, 6},
    {"_tinytorch_C_torch_max_pool1d", (DL_FUNC) &_tinytorch_C_torch_max_pool1d, 6},
    {"_tinytorch_C_torch_max_pool2d", (DL_FUNC) &_tinytorch_C_torch_max_pool2d, 6},
    {"_tinytorch_C_torch_quantized_max_pool1d", (DL_FUNC) &_tinytorch_C_torch_quantized_max_pool1d, 6},
    {"_tinytorch_C_torch_quantized_max_pool2d", (DL_FUNC) &_tinytorch_C_torch_quantized_max_pool2d, 6},
    {"_tinytorch_C_torch_quantized_max_pool3d", (DL_FUNC) &_tinytorch_C_torch_quantized_max_pool3d, 6},
    {"_tinytorch_C_torch_max_pool3d", (DL_FUNC) &_tinytorch_C_torch_max_pool3d, 6},
    {"_tinytorch_C_torch_nanmean", (DL_FUNC) &_tinytorch_C_torch_nanmean, 4},
    {"_tinytorch_C_torch_median", (DL_FUNC) &_tinytorch_C_torch_median, 1},
    {"_tinytorch_C_torch_nanmedian", (DL_FUNC) &_tinytorch_C_torch_nanmedian, 1},
    {"_tinytorch_C_torch_amin", (DL_FUNC) &_tinytorch_C_torch_amin, 3},
    {"_tinytorch_C_torch_mode", (DL_FUNC) &_tinytorch_C_torch_mode, 3},
    {"_tinytorch_C_torch_mul_", (DL_FUNC) &_tinytorch_C_torch_mul_, 2},
    {"_tinytorch_C_torch_multiply", (DL_FUNC) &_tinytorch_C_torch_multiply, 2},
    {"_tinytorch_C_torch_multiply_", (DL_FUNC) &_tinytorch_C_torch_multiply_, 2},
    {"_tinytorch_C_torch_mv", (DL_FUNC) &_tinytorch_C_torch_mv, 2},
    {"_tinytorch_C_torch_mvlgamma", (DL_FUNC) &_tinytorch_C_torch_mvlgamma, 2},
    {"_tinytorch_C_torch_mvlgamma_", (DL_FUNC) &_tinytorch_C_torch_mvlgamma_, 2},
    {"_tinytorch_C_torch_native_batch_norm", (DL_FUNC) &_tinytorch_C_torch_native_batch_norm, 8},
    {"_tinytorch_C_torch_batch_norm_stats", (DL_FUNC) &_tinytorch_C_torch_batch_norm_stats, 2},
    {"_tinytorch_C_torch_batch_norm_elemt", (DL_FUNC) &_tinytorch_C_torch_batch_norm_elemt, 6},
    {"_tinytorch_C_torch_batch_norm_gather_stats", (DL_FUNC) &_tinytorch_C_torch_batch_norm_gather_stats, 8},
    {"_tinytorch_C_torch_batch_norm_gather_stats_with_counts", (DL_FUNC) &_tinytorch_C_torch_batch_norm_gather_stats_with_counts, 8},
    {"_tinytorch_C_torch_batch_norm_update_stats", (DL_FUNC) &_tinytorch_C_torch_batch_norm_update_stats, 4},
    {"_tinytorch_C_torch_is_vulkan_available", (DL_FUNC) &_tinytorch_C_torch_is_vulkan_available, 0},
    {"_tinytorch_C_torch_pairwise_distance", (DL_FUNC) &_tinytorch_C_torch_pairwise_distance, 5},
    {"_tinytorch_C_torch_cdist", (DL_FUNC) &_tinytorch_C_torch_cdist, 4},
    {"_tinytorch_C_torch_pdist", (DL_FUNC) &_tinytorch_C_torch_pdist, 2},
    {"_tinytorch_C_torch_cosine_similarity", (DL_FUNC) &_tinytorch_C_torch_cosine_similarity, 4},
    {"_tinytorch_C_torch_movedim", (DL_FUNC) &_tinytorch_C_torch_movedim, 3},
    {"_tinytorch_C_torch_moveaxis", (DL_FUNC) &_tinytorch_C_torch_moveaxis, 3},
    {"_tinytorch_C_torch_numpy_T", (DL_FUNC) &_tinytorch_C_torch_numpy_T, 1},
    {"_tinytorch_C_torch_matrix_H", (DL_FUNC) &_tinytorch_C_torch_matrix_H, 1},
    {"_tinytorch_C_torch_mT", (DL_FUNC) &_tinytorch_C_torch_mT, 1},
    {"_tinytorch_C_torch_mH", (DL_FUNC) &_tinytorch_C_torch_mH, 1},
    {"_tinytorch_C_torch_adjoint", (DL_FUNC) &_tinytorch_C_torch_adjoint, 1},
    {"_tinytorch_C_torch_pixel_shuffle", (DL_FUNC) &_tinytorch_C_torch_pixel_shuffle, 2},
    {"_tinytorch_C_torch_pixel_unshuffle", (DL_FUNC) &_tinytorch_C_torch_pixel_unshuffle, 2},
    {"_tinytorch_C_torch_channel_shuffle", (DL_FUNC) &_tinytorch_C_torch_channel_shuffle, 2},
    {"_tinytorch_C_torch_native_channel_shuffle", (DL_FUNC) &_tinytorch_C_torch_native_channel_shuffle, 2},
    {"_tinytorch_C_torch_is_pinned", (DL_FUNC) &_tinytorch_C_torch_is_pinned, 2},
    {"_tinytorch_C_torch_pin_memory", (DL_FUNC) &_tinytorch_C_torch_pin_memory, 2},
    {"_tinytorch_C_torch_pinverse", (DL_FUNC) &_tinytorch_C_torch_pinverse, 2},
    {"_tinytorch_C_torch_poisson_nll_loss", (DL_FUNC) &_tinytorch_C_torch_poisson_nll_loss, 6},
    {"_tinytorch_C_torch_rad2deg", (DL_FUNC) &_tinytorch_C_torch_rad2deg, 1},
    {"_tinytorch_C_torch_rad2deg_", (DL_FUNC) &_tinytorch_C_torch_rad2deg_, 1},
    {"_tinytorch_C_torch_deg2rad", (DL_FUNC) &_tinytorch_C_torch_deg2rad, 1},
    {"_tinytorch_C_torch_deg2rad_", (DL_FUNC) &_tinytorch_C_torch_deg2rad_, 1},
    {"_tinytorch_C_torch_scalar_tensor", (DL_FUNC) &_tinytorch_C_torch_scalar_tensor, 3},
    {"_tinytorch_C_torch_rand", (DL_FUNC) &_tinytorch_C_torch_rand, 4},
    {"_tinytorch_C_torch_rand_like", (DL_FUNC) &_tinytorch_C_torch_rand_like, 4},
    {"_tinytorch_C_torch_randint", (DL_FUNC) &_tinytorch_C_torch_randint, 4},
    {"_tinytorch_C_torch_randint_like", (DL_FUNC) &_tinytorch_C_torch_randint_like, 5},
    {"_tinytorch_C_torch_randperm", (DL_FUNC) &_tinytorch_C_torch_randperm, 3},
    {"_tinytorch_C_torch_range", (DL_FUNC) &_tinytorch_C_torch_range, 5},
    {"_tinytorch_C_torch_ravel", (DL_FUNC) &_tinytorch_C_torch_ravel, 1},
    {"_tinytorch_C_torch_reciprocal", (DL_FUNC) &_tinytorch_C_torch_reciprocal, 1},
    {"_tinytorch_C_torch_reciprocal_", (DL_FUNC) &_tinytorch_C_torch_reciprocal_, 1},
    {"_tinytorch_C_torch_neg_", (DL_FUNC) &_tinytorch_C_torch_neg_, 1},
    {"_tinytorch_C_torch_negative", (DL_FUNC) &_tinytorch_C_torch_negative, 1},
    {"_tinytorch_C_torch_negative_", (DL_FUNC) &_tinytorch_C_torch_negative_, 1},
    {"_tinytorch_C_torch_reshape_as", (DL_FUNC) &_tinytorch_C_torch_reshape_as, 2},
    {"_tinytorch_C_torch_round_", (DL_FUNC) &_tinytorch_C_torch_round_, 1},
    {"_tinytorch_C_torch_rrelu", (DL_FUNC) &_tinytorch_C_torch_rrelu, 5},
    {"_tinytorch_C_torch_rrelu_", (DL_FUNC) &_tinytorch_C_torch_rrelu_, 5},
    {"_tinytorch_C_torch_relu_", (DL_FUNC) &_tinytorch_C_torch_relu_, 1},
    {"_tinytorch_C_torch_relu6", (DL_FUNC) &_tinytorch_C_torch_relu6, 1},
    {"_tinytorch_C_torch_relu6_", (DL_FUNC) &_tinytorch_C_torch_relu6_, 1},
    {"_tinytorch_C_torch_prelu", (DL_FUNC) &_tinytorch_C_torch_prelu, 2},
    {"_tinytorch_C_torch_gelu_", (DL_FUNC) &_tinytorch_C_torch_gelu_, 2},
    {"_tinytorch_C_torch_hardshrink", (DL_FUNC) &_tinytorch_C_torch_hardshrink, 2},
    {"_tinytorch_C_torch_rsqrt_", (DL_FUNC) &_tinytorch_C_torch_rsqrt_, 1},
    {"_tinytorch_C_torch_select", (DL_FUNC) &_tinytorch_C_torch_select, 3},
    {"_tinytorch_C_torch_selu", (DL_FUNC) &_tinytorch_C_torch_selu, 1},
    {"_tinytorch_C_torch_selu_", (DL_FUNC) &_tinytorch_C_torch_selu_, 1},
    {"_tinytorch_C_torch_celu", (DL_FUNC) &_tinytorch_C_torch_celu, 2},
    {"_tinytorch_C_torch_celu_", (DL_FUNC) &_tinytorch_C_torch_celu_, 2},
    {"_tinytorch_C_torch_silu_", (DL_FUNC) &_tinytorch_C_torch_silu_, 1},
    {"_tinytorch_C_torch_mish", (DL_FUNC) &_tinytorch_C_torch_mish, 1},
    {"_tinytorch_C_torch_mish_", (DL_FUNC) &_tinytorch_C_torch_mish_, 1},
    {"_tinytorch_C_torch_sigmoid_", (DL_FUNC) &_tinytorch_C_torch_sigmoid_, 1},
    {"_tinytorch_C_torch_logit", (DL_FUNC) &_tinytorch_C_torch_logit, 2},
    {"_tinytorch_C_torch_logit_", (DL_FUNC) &_tinytorch_C_torch_logit_, 2},
    {"_tinytorch_C_torch_sin_", (DL_FUNC) &_tinytorch_C_torch_sin_, 1},
    {"_tinytorch_C_torch_sinc", (DL_FUNC) &_tinytorch_C_torch_sinc, 1},
    {"_tinytorch_C_torch_sinc_", (DL_FUNC) &_tinytorch_C_torch_sinc_, 1},
    {"_tinytorch_C_torch_sinh", (DL_FUNC) &_tinytorch_C_torch_sinh, 1},
    {"_tinytorch_C_torch_sinh_", (DL_FUNC) &_tinytorch_C_torch_sinh_, 1},
    {"_tinytorch_C_torch_detach_", (DL_FUNC) &_tinytorch_C_torch_detach_, 1},
    {"_tinytorch_C_torch_slice", (DL_FUNC) &_tinytorch_C_torch_slice, 5},
    {"_tinytorch_C_torch_slice_inverse", (DL_FUNC) &_tinytorch_C_torch_slice_inverse, 6},
    {"_tinytorch_C_torch_slice_scatter", (DL_FUNC) &_tinytorch_C_torch_slice_scatter, 6},
    {"_tinytorch_C_torch_select_scatter", (DL_FUNC) &_tinytorch_C_torch_select_scatter, 4},
    {"_tinytorch_C_torch_diagonal_scatter", (DL_FUNC) &_tinytorch_C_torch_diagonal_scatter, 5},
    {"_tinytorch_C_torch_as_strided_scatter", (DL_FUNC) &_tinytorch_C_torch_as_strided_scatter, 5},
    {"_tinytorch_C_torch_smm", (DL_FUNC) &_tinytorch_C_torch_smm, 2},
    {"_tinytorch_C_torch_unsafe_split", (DL_FUNC) &_tinytorch_C_torch_unsafe_split, 3},
    {"_tinytorch_C_torch_split", (DL_FUNC) &_tinytorch_C_torch_split, 3},
    {"_tinytorch_C_torch_unsafe_split_with_sizes", (DL_FUNC) &_tinytorch_C_torch_unsafe_split_with_sizes, 3},
    {"_tinytorch_C_torch_split_with_sizes", (DL_FUNC) &_tinytorch_C_torch_split_with_sizes, 3},
    {"_tinytorch_C_torch_hsplit", (DL_FUNC) &_tinytorch_C_torch_hsplit, 2},
    {"_tinytorch_C_torch_vsplit", (DL_FUNC) &_tinytorch_C_torch_vsplit, 2},
    {"_tinytorch_C_torch_dsplit", (DL_FUNC) &_tinytorch_C_torch_dsplit, 2},
    {"_tinytorch_C_torch_squeeze_", (DL_FUNC) &_tinytorch_C_torch_squeeze_, 1},
    {"_tinytorch_C_torch_sspaddmm", (DL_FUNC) &_tinytorch_C_torch_sspaddmm, 5},
    {"_tinytorch_C_torch_stack", (DL_FUNC) &_tinytorch_C_torch_stack, 2},
    {"_tinytorch_C_torch_hstack", (DL_FUNC) &_tinytorch_C_torch_hstack, 1},
    {"_tinytorch_C_torch_vstack", (DL_FUNC) &_tinytorch_C_torch_vstack, 1},
    {"_tinytorch_C_torch_dstack", (DL_FUNC) &_tinytorch_C_torch_dstack, 1},
    {"_tinytorch_C_torch_stride", (DL_FUNC) &_tinytorch_C_torch_stride, 2},
    {"_tinytorch_C_torch_nansum", (DL_FUNC) &_tinytorch_C_torch_nansum, 4},
    {"_tinytorch_C_torch_sum_to_size", (DL_FUNC) &_tinytorch_C_torch_sum_to_size, 2},
    {"_tinytorch_C_torch_sqrt_", (DL_FUNC) &_tinytorch_C_torch_sqrt_, 1},
    {"_tinytorch_C_torch_square", (DL_FUNC) &_tinytorch_C_torch_square, 1},
    {"_tinytorch_C_torch_square_", (DL_FUNC) &_tinytorch_C_torch_square_, 1},
    {"_tinytorch_C_torch_std_mean", (DL_FUNC) &_tinytorch_C_torch_std_mean, 2},
    {"_tinytorch_C_torch_prod", (DL_FUNC) &_tinytorch_C_torch_prod, 2},
    {"_tinytorch_C_torch_t_", (DL_FUNC) &_tinytorch_C_torch_t_, 1},
    {"_tinytorch_C_torch_tan", (DL_FUNC) &_tinytorch_C_torch_tan, 1},
    {"_tinytorch_C_torch_tan_", (DL_FUNC) &_tinytorch_C_torch_tan_, 1},
    {"_tinytorch_C_torch_tanh_", (DL_FUNC) &_tinytorch_C_torch_tanh_, 1},
    {"_tinytorch_C_torch_tensordot", (DL_FUNC) &_tinytorch_C_torch_tensordot, 4},
    {"_tinytorch_C_torch_threshold", (DL_FUNC) &_tinytorch_C_torch_threshold, 3},
    {"_tinytorch_C_torch_threshold_", (DL_FUNC) &_tinytorch_C_torch_threshold_, 3},
    {"_tinytorch_C_torch_tile", (DL_FUNC) &_tinytorch_C_torch_tile, 2},
    {"_tinytorch_C_torch_transpose_", (DL_FUNC) &_tinytorch_C_torch_transpose_, 3},
    {"_tinytorch_C_torch_one_hot", (DL_FUNC) &_tinytorch_C_torch_one_hot, 2},
    {"_tinytorch_C_torch_fliplr", (DL_FUNC) &_tinytorch_C_torch_fliplr, 1},
    {"_tinytorch_C_torch_flipud", (DL_FUNC) &_tinytorch_C_torch_flipud, 1},
    {"_tinytorch_C_torch_roll", (DL_FUNC) &_tinytorch_C_torch_roll, 3},
    {"_tinytorch_C_torch_rot90", (DL_FUNC) &_tinytorch_C_torch_rot90, 3},
    {"_tinytorch_C_torch_trapezoid", (DL_FUNC) &_tinytorch_C_torch_trapezoid, 3},
    {"_tinytorch_C_torch_trapz", (DL_FUNC) &_tinytorch_C_torch_trapz, 3},
    {"_tinytorch_C_torch_triplet_margin_loss", (DL_FUNC) &_tinytorch_C_torch_triplet_margin_loss, 8},
    {"_tinytorch_C_torch_trunc_", (DL_FUNC) &_tinytorch_C_torch_trunc_, 1},
    {"_tinytorch_C_torch_fix", (DL_FUNC) &_tinytorch_C_torch_fix, 1},
    {"_tinytorch_C_torch_fix_", (DL_FUNC) &_tinytorch_C_torch_fix_, 1},
    {"_tinytorch_C_torch_type_as", (DL_FUNC) &_tinytorch_C_torch_type_as, 2},
    {"_tinytorch_C_torch_unique_dim", (DL_FUNC) &_tinytorch_C_torch_unique_dim, 5},
    {"_tinytorch_C_torch_unique_consecutive", (DL_FUNC) &_tinytorch_C_torch_unique_consecutive, 4},
    {"_tinytorch_C_torch_unique_dim_consecutive", (DL_FUNC) &_tinytorch_C_torch_unique_dim_consecutive, 4},
    {"_tinytorch_C_torch_unsqueeze_", (DL_FUNC) &_tinytorch_C_torch_unsqueeze_, 2},
    {"_tinytorch_C_torch_vander", (DL_FUNC) &_tinytorch_C_torch_vander, 3},
    {"_tinytorch_C_torch_var", (DL_FUNC) &_tinytorch_C_torch_var, 2},
    {"_tinytorch_C_torch_var_mean", (DL_FUNC) &_tinytorch_C_torch_var_mean, 2},
    {"_tinytorch_C_torch_view_as", (DL_FUNC) &_tinytorch_C_torch_view_as, 2},
    {"_tinytorch_C_torch_norm_except_dim", (DL_FUNC) &_tinytorch_C_torch_norm_except_dim, 3},
    {"_tinytorch_C_torch_poisson", (DL_FUNC) &_tinytorch_C_torch_poisson, 2},
    {"_tinytorch_C_torch_binomial", (DL_FUNC) &_tinytorch_C_torch_binomial, 3},
    {"_tinytorch_C_torch_native_norm", (DL_FUNC) &_tinytorch_C_torch_native_norm, 2},
    {"_tinytorch_C_torch_frexp", (DL_FUNC) &_tinytorch_C_torch_frexp, 1},
    {"_tinytorch_C_torch_frobenius_norm", (DL_FUNC) &_tinytorch_C_torch_frobenius_norm, 3},
    {"_tinytorch_C_torch_nuclear_norm", (DL_FUNC) &_tinytorch_C_torch_nuclear_norm, 2},
    {"_tinytorch_C_torch_positive", (DL_FUNC) &_tinytorch_C_torch_positive, 1},
    {"_tinytorch_C_torch_resize_as_", (DL_FUNC) &_tinytorch_C_torch_resize_as_, 3},
    {"_tinytorch_C_torch_resize_as_sparse_", (DL_FUNC) &_tinytorch_C_torch_resize_as_sparse_, 2},
    {"_tinytorch_C_torch_sub_", (DL_FUNC) &_tinytorch_C_torch_sub_, 3},
    {"_tinytorch_C_torch_subtract", (DL_FUNC) &_tinytorch_C_torch_subtract, 3},
    {"_tinytorch_C_torch_subtract_", (DL_FUNC) &_tinytorch_C_torch_subtract_, 3},
    {"_tinytorch_C_torch_rsub", (DL_FUNC) &_tinytorch_C_torch_rsub, 3},
    {"_tinytorch_C_torch_heaviside", (DL_FUNC) &_tinytorch_C_torch_heaviside, 2},
    {"_tinytorch_C_torch_heaviside_", (DL_FUNC) &_tinytorch_C_torch_heaviside_, 2},
    {"_tinytorch_C_torch_sparse_sampled_addmm", (DL_FUNC) &_tinytorch_C_torch_sparse_sampled_addmm, 5},
    {"_tinytorch_C_torch_addmm", (DL_FUNC) &_tinytorch_C_torch_addmm, 5},
    {"_tinytorch_C_torch_addmm_", (DL_FUNC) &_tinytorch_C_torch_addmm_, 5},
    {"_tinytorch_C_torch_sparse_compressed_tensor", (DL_FUNC) &_tinytorch_C_torch_sparse_compressed_tensor, 6},
    {"_tinytorch_C_torch_sparse_csr_tensor", (DL_FUNC) &_tinytorch_C_torch_sparse_csr_tensor, 6},
    {"_tinytorch_C_torch_sparse_csc_tensor", (DL_FUNC) &_tinytorch_C_torch_sparse_csc_tensor, 6},
    {"_tinytorch_C_torch_sparse_bsr_tensor", (DL_FUNC) &_tinytorch_C_torch_sparse_bsr_tensor, 6},
    {"_tinytorch_C_torch_sparse_bsc_tensor", (DL_FUNC) &_tinytorch_C_torch_sparse_bsc_tensor, 6},
    {"_tinytorch_C_torch_sparse_coo_tensor", (DL_FUNC) &_tinytorch_C_torch_sparse_coo_tensor, 3},
    {"_tinytorch_C_torch_sparse_resize_", (DL_FUNC) &_tinytorch_C_torch_sparse_resize_, 4},
    {"_tinytorch_C_torch_sparse_resize_and_clear_", (DL_FUNC) &_tinytorch_C_torch_sparse_resize_and_clear_, 4},
    {"_tinytorch_C_torch_sparse_mask", (DL_FUNC) &_tinytorch_C_torch_sparse_mask, 2},
    {"_tinytorch_C_torch_to_dense", (DL_FUNC) &_tinytorch_C_torch_to_dense, 3},
    {"_tinytorch_C_torch_sparse_dim", (DL_FUNC) &_tinytorch_C_torch_sparse_dim, 1},
    {"_tinytorch_C_torch_dense_dim", (DL_FUNC) &_tinytorch_C_torch_dense_dim, 1},
    {"_tinytorch_C_torch_coalesce", (DL_FUNC) &_tinytorch_C_torch_coalesce, 1},
    {"_tinytorch_C_torch_is_coalesced", (DL_FUNC) &_tinytorch_C_torch_is_coalesced, 1},
    {"_tinytorch_C_torch_indices", (DL_FUNC) &_tinytorch_C_torch_indices, 1},
    {"_tinytorch_C_torch_values", (DL_FUNC) &_tinytorch_C_torch_values, 1},
    {"_tinytorch_C_torch_crow_indices", (DL_FUNC) &_tinytorch_C_torch_crow_indices, 1},
    {"_tinytorch_C_torch_col_indices", (DL_FUNC) &_tinytorch_C_torch_col_indices, 1},
    {"_tinytorch_C_torch_ccol_indices", (DL_FUNC) &_tinytorch_C_torch_ccol_indices, 1},
    {"_tinytorch_C_torch_row_indices", (DL_FUNC) &_tinytorch_C_torch_row_indices, 1},
    {"_tinytorch_C_torch_hspmm", (DL_FUNC) &_tinytorch_C_torch_hspmm, 2},
    {"_tinytorch_C_torch_copy_sparse_to_sparse_", (DL_FUNC) &_tinytorch_C_torch_copy_sparse_to_sparse_, 3},
    {"_tinytorch_C_torch_unbind", (DL_FUNC) &_tinytorch_C_torch_unbind, 2},
    {"_tinytorch_C_torch_to_sparse", (DL_FUNC) &_tinytorch_C_torch_to_sparse, 2},
    {"_tinytorch_C_torch_to_sparse_csr", (DL_FUNC) &_tinytorch_C_torch_to_sparse_csr, 2},
    {"_tinytorch_C_torch_to_sparse_csc", (DL_FUNC) &_tinytorch_C_torch_to_sparse_csc, 2},
    {"_tinytorch_C_torch_to_sparse_bsr", (DL_FUNC) &_tinytorch_C_torch_to_sparse_bsr, 3},
    {"_tinytorch_C_torch_to_sparse_bsc", (DL_FUNC) &_tinytorch_C_torch_to_sparse_bsc, 3},
    {"_tinytorch_C_torch_quantize_per_tensor_dynamic", (DL_FUNC) &_tinytorch_C_torch_quantize_per_tensor_dynamic, 3},
    {"_tinytorch_C_torch_quantize_per_tensor", (DL_FUNC) &_tinytorch_C_torch_quantize_per_tensor, 4},
    {"_tinytorch_C_torch_quantize_per_channel", (DL_FUNC) &_tinytorch_C_torch_quantize_per_channel, 5},
    {"_tinytorch_C_torch_dequantize", (DL_FUNC) &_tinytorch_C_torch_dequantize, 1},
    {"_tinytorch_C_torch_q_scale", (DL_FUNC) &_tinytorch_C_torch_q_scale, 1},
    {"_tinytorch_C_torch_q_zero_point", (DL_FUNC) &_tinytorch_C_torch_q_zero_point, 1},
    {"_tinytorch_C_torch_q_per_channel_scales", (DL_FUNC) &_tinytorch_C_torch_q_per_channel_scales, 1},
    {"_tinytorch_C_torch_q_per_channel_zero_points", (DL_FUNC) &_tinytorch_C_torch_q_per_channel_zero_points, 1},
    {"_tinytorch_C_torch_q_per_channel_axis", (DL_FUNC) &_tinytorch_C_torch_q_per_channel_axis, 1},
    {"_tinytorch_C_torch_int_repr", (DL_FUNC) &_tinytorch_C_torch_int_repr, 1},
    {"_tinytorch_C_torch_qscheme", (DL_FUNC) &_tinytorch_C_torch_qscheme, 1},
    {"_tinytorch_C_torch_fake_quantize_per_tensor_affine", (DL_FUNC) &_tinytorch_C_torch_fake_quantize_per_tensor_affine, 5},
    {"_tinytorch_C_torch_fake_quantize_per_tensor_affine_cachemask", (DL_FUNC) &_tinytorch_C_torch_fake_quantize_per_tensor_affine_cachemask, 5},
    {"_tinytorch_C_torch_fake_quantize_per_channel_affine", (DL_FUNC) &_tinytorch_C_torch_fake_quantize_per_channel_affine, 6},
    {"_tinytorch_C_torch_fake_quantize_per_channel_affine_cachemask", (DL_FUNC) &_tinytorch_C_torch_fake_quantize_per_channel_affine_cachemask, 6},
    {"_tinytorch_C_torch_fused_moving_avg_obs_fake_quant", (DL_FUNC) &_tinytorch_C_torch_fused_moving_avg_obs_fake_quant, 13},
    {"_tinytorch_C_torch_choose_qparams_optimized", (DL_FUNC) &_tinytorch_C_torch_choose_qparams_optimized, 5},
    {"_tinytorch_C_torch_meshgrid", (DL_FUNC) &_tinytorch_C_torch_meshgrid, 1},
    {"_tinytorch_C_torch_cartesian_prod", (DL_FUNC) &_tinytorch_C_torch_cartesian_prod, 1},
    {"_tinytorch_C_torch_combinations", (DL_FUNC) &_tinytorch_C_torch_combinations, 3},
    {"_tinytorch_C_torch_result_type", (DL_FUNC) &_tinytorch_C_torch_result_type, 2},
    {"_tinytorch_C_torch_can_cast", (DL_FUNC) &_tinytorch_C_torch_can_cast, 2},
    {"_tinytorch_C_torch_promote_types", (DL_FUNC) &_tinytorch_C_torch_promote_types, 2},
    {"_tinytorch_C_torch_gru", (DL_FUNC) &_tinytorch_C_torch_gru, 9},
    {"_tinytorch_C_torch_rnn_tanh", (DL_FUNC) &_tinytorch_C_torch_rnn_tanh, 9},
    {"_tinytorch_C_torch_rnn_relu", (DL_FUNC) &_tinytorch_C_torch_rnn_relu, 9},
    {"_tinytorch_C_torch_lstm_cell", (DL_FUNC) &_tinytorch_C_torch_lstm_cell, 6},
    {"_tinytorch_C_torch_gru_cell", (DL_FUNC) &_tinytorch_C_torch_gru_cell, 6},
    {"_tinytorch_C_torch_rnn_tanh_cell", (DL_FUNC) &_tinytorch_C_torch_rnn_tanh_cell, 6},
    {"_tinytorch_C_torch_rnn_relu_cell", (DL_FUNC) &_tinytorch_C_torch_rnn_relu_cell, 6},
    {"_tinytorch_C_torch_quantized_lstm_cell", (DL_FUNC) &_tinytorch_C_torch_quantized_lstm_cell, 14},
    {"_tinytorch_C_torch_quantized_gru_cell", (DL_FUNC) &_tinytorch_C_torch_quantized_gru_cell, 14},
    {"_tinytorch_C_torch_quantized_rnn_relu_cell", (DL_FUNC) &_tinytorch_C_torch_quantized_rnn_relu_cell, 14},
    {"_tinytorch_C_torch_quantized_rnn_tanh_cell", (DL_FUNC) &_tinytorch_C_torch_quantized_rnn_tanh_cell, 14},
    {"_tinytorch_C_torch_set_", (DL_FUNC) &_tinytorch_C_torch_set_, 5},
    {"_tinytorch_C_torch_is_set_to", (DL_FUNC) &_tinytorch_C_torch_is_set_to, 2},
    {"_tinytorch_C_torch_masked_scatter_", (DL_FUNC) &_tinytorch_C_torch_masked_scatter_, 3},
    {"_tinytorch_C_torch_masked_scatter", (DL_FUNC) &_tinytorch_C_torch_masked_scatter, 3},
    {"_tinytorch_C_torch_put_", (DL_FUNC) &_tinytorch_C_torch_put_, 4},
    {"_tinytorch_C_torch_put", (DL_FUNC) &_tinytorch_C_torch_put, 4},
    {"_tinytorch_C_torch_index_add_", (DL_FUNC) &_tinytorch_C_torch_index_add_, 5},
    {"_tinytorch_C_torch_index_add", (DL_FUNC) &_tinytorch_C_torch_index_add, 5},
    {"_tinytorch_C_torch_index_reduce_", (DL_FUNC) &_tinytorch_C_torch_index_reduce_, 6},
    {"_tinytorch_C_torch_index_reduce", (DL_FUNC) &_tinytorch_C_torch_index_reduce, 6},
    {"_tinytorch_C_torch_index_fill_", (DL_FUNC) &_tinytorch_C_torch_index_fill_, 4},
    {"_tinytorch_C_torch_index_fill", (DL_FUNC) &_tinytorch_C_torch_index_fill, 4},
    {"_tinytorch_C_torch_scatter", (DL_FUNC) &_tinytorch_C_torch_scatter, 4},
    {"_tinytorch_C_torch_scatter_add", (DL_FUNC) &_tinytorch_C_torch_scatter_add, 4},
    {"_tinytorch_C_torch_scatter_add_", (DL_FUNC) &_tinytorch_C_torch_scatter_add_, 4},
    {"_tinytorch_C_torch_scatter_reduce", (DL_FUNC) &_tinytorch_C_torch_scatter_reduce, 6},
    {"_tinytorch_C_torch_scatter_reduce_", (DL_FUNC) &_tinytorch_C_torch_scatter_reduce_, 6},
    {"_tinytorch_C_torch_eq_", (DL_FUNC) &_tinytorch_C_torch_eq_, 2},
    {"_tinytorch_C_torch_bitwise_and", (DL_FUNC) &_tinytorch_C_torch_bitwise_and, 2},
    {"_tinytorch_C_torch_bitwise_and_", (DL_FUNC) &_tinytorch_C_torch_bitwise_and_, 2},
    {"_tinytorch_C_torch___and__", (DL_FUNC) &_tinytorch_C_torch___and__, 2},
    {"_tinytorch_C_torch___iand__", (DL_FUNC) &_tinytorch_C_torch___iand__, 2},
    {"_tinytorch_C_torch_bitwise_or", (DL_FUNC) &_tinytorch_C_torch_bitwise_or, 2},
    {"_tinytorch_C_torch_bitwise_or_", (DL_FUNC) &_tinytorch_C_torch_bitwise_or_, 2},
    {"_tinytorch_C_torch___or__", (DL_FUNC) &_tinytorch_C_torch___or__, 2},
    {"_tinytorch_C_torch___ior__", (DL_FUNC) &_tinytorch_C_torch___ior__, 2},
    {"_tinytorch_C_torch_bitwise_xor", (DL_FUNC) &_tinytorch_C_torch_bitwise_xor, 2},
    {"_tinytorch_C_torch_bitwise_xor_", (DL_FUNC) &_tinytorch_C_torch_bitwise_xor_, 2},
    {"_tinytorch_C_torch___xor__", (DL_FUNC) &_tinytorch_C_torch___xor__, 2},
    {"_tinytorch_C_torch___ixor__", (DL_FUNC) &_tinytorch_C_torch___ixor__, 2},
    {"_tinytorch_C_torch___lshift__", (DL_FUNC) &_tinytorch_C_torch___lshift__, 2},
    {"_tinytorch_C_torch___ilshift__", (DL_FUNC) &_tinytorch_C_torch___ilshift__, 2},
    {"_tinytorch_C_torch_bitwise_left_shift", (DL_FUNC) &_tinytorch_C_torch_bitwise_left_shift, 2},
    {"_tinytorch_C_torch_bitwise_left_shift_", (DL_FUNC) &_tinytorch_C_torch_bitwise_left_shift_, 2},
    {"_tinytorch_C_torch___rshift__", (DL_FUNC) &_tinytorch_C_torch___rshift__, 2},
    {"_tinytorch_C_torch___irshift__", (DL_FUNC) &_tinytorch_C_torch___irshift__, 2},
    {"_tinytorch_C_torch_bitwise_right_shift", (DL_FUNC) &_tinytorch_C_torch_bitwise_right_shift, 2},
    {"_tinytorch_C_torch_bitwise_right_shift_", (DL_FUNC) &_tinytorch_C_torch_bitwise_right_shift_, 2},
    {"_tinytorch_C_torch_tril_", (DL_FUNC) &_tinytorch_C_torch_tril_, 2},
    {"_tinytorch_C_torch_triu_", (DL_FUNC) &_tinytorch_C_torch_triu_, 2},
    {"_tinytorch_C_torch_digamma_", (DL_FUNC) &_tinytorch_C_torch_digamma_, 1},
    {"_tinytorch_C_torch_lerp_", (DL_FUNC) &_tinytorch_C_torch_lerp_, 3},
    {"_tinytorch_C_torch_addbmm_", (DL_FUNC) &_tinytorch_C_torch_addbmm_, 5},
    {"_tinytorch_C_torch_addbmm", (DL_FUNC) &_tinytorch_C_torch_addbmm, 5},
    {"_tinytorch_C_torch_random_", (DL_FUNC) &_tinytorch_C_torch_random_, 4},
    {"_tinytorch_C_torch_cauchy_", (DL_FUNC) &_tinytorch_C_torch_cauchy_, 4},
    {"_tinytorch_C_torch_log_normal_", (DL_FUNC) &_tinytorch_C_torch_log_normal_, 4},
    {"_tinytorch_C_torch_exponential_", (DL_FUNC) &_tinytorch_C_torch_exponential_, 3},
    {"_tinytorch_C_torch_geometric_", (DL_FUNC) &_tinytorch_C_torch_geometric_, 3},
    {"_tinytorch_C_torch_diag", (DL_FUNC) &_tinytorch_C_torch_diag, 2},
    {"_tinytorch_C_torch_cross", (DL_FUNC) &_tinytorch_C_torch_cross, 3},
    {"_tinytorch_C_torch_tril", (DL_FUNC) &_tinytorch_C_torch_tril, 2},
    {"_tinytorch_C_torch_tril_indices", (DL_FUNC) &_tinytorch_C_torch_tril_indices, 5},
    {"_tinytorch_C_torch_triu_indices", (DL_FUNC) &_tinytorch_C_torch_triu_indices, 5},
    {"_tinytorch_C_torch_trace", (DL_FUNC) &_tinytorch_C_torch_trace, 1},
    {"_tinytorch_C_torch_ne_", (DL_FUNC) &_tinytorch_C_torch_ne_, 2},
    {"_tinytorch_C_torch_not_equal", (DL_FUNC) &_tinytorch_C_torch_not_equal, 2},
    {"_tinytorch_C_torch_not_equal_", (DL_FUNC) &_tinytorch_C_torch_not_equal_, 2},
    {"_tinytorch_C_torch_ge_", (DL_FUNC) &_tinytorch_C_torch_ge_, 2},
    {"_tinytorch_C_torch_greater_equal", (DL_FUNC) &_tinytorch_C_torch_greater_equal, 2},
    {"_tinytorch_C_torch_greater_equal_", (DL_FUNC) &_tinytorch_C_torch_greater_equal_, 2},
    {"_tinytorch_C_torch_le_", (DL_FUNC) &_tinytorch_C_torch_le_, 2},
    {"_tinytorch_C_torch_less_equal", (DL_FUNC) &_tinytorch_C_torch_less_equal, 2},
    {"_tinytorch_C_torch_less_equal_", (DL_FUNC) &_tinytorch_C_torch_less_equal_, 2},
    {"_tinytorch_C_torch_gt_", (DL_FUNC) &_tinytorch_C_torch_gt_, 2},
    {"_tinytorch_C_torch_greater", (DL_FUNC) &_tinytorch_C_torch_greater, 2},
    {"_tinytorch_C_torch_greater_", (DL_FUNC) &_tinytorch_C_torch_greater_, 2},
    {"_tinytorch_C_torch_lt_", (DL_FUNC) &_tinytorch_C_torch_lt_, 2},
    {"_tinytorch_C_torch_less", (DL_FUNC) &_tinytorch_C_torch_less, 2},
    {"_tinytorch_C_torch_less_", (DL_FUNC) &_tinytorch_C_torch_less_, 2},
    {"_tinytorch_C_torch_take", (DL_FUNC) &_tinytorch_C_torch_take, 2},
    {"_tinytorch_C_torch_take_along_dim", (DL_FUNC) &_tinytorch_C_torch_take_along_dim, 3},
    {"_tinytorch_C_torch_masked_select", (DL_FUNC) &_tinytorch_C_torch_masked_select, 2},
    {"_tinytorch_C_torch_nonzero", (DL_FUNC) &_tinytorch_C_torch_nonzero, 1},
    {"_tinytorch_C_torch_nonzero_static", (DL_FUNC) &_tinytorch_C_torch_nonzero_static, 3},
    {"_tinytorch_C_torch_nonzero_numpy", (DL_FUNC) &_tinytorch_C_torch_nonzero_numpy, 1},
    {"_tinytorch_C_torch_argwhere", (DL_FUNC) &_tinytorch_C_torch_argwhere, 1},
    {"_tinytorch_C_torch_addcmul", (DL_FUNC) &_tinytorch_C_torch_addcmul, 4},
    {"_tinytorch_C_torch_addcmul_", (DL_FUNC) &_tinytorch_C_torch_addcmul_, 4},
    {"_tinytorch_C_torch_addcdiv", (DL_FUNC) &_tinytorch_C_torch_addcdiv, 4},
    {"_tinytorch_C_torch_addcdiv_", (DL_FUNC) &_tinytorch_C_torch_addcdiv_, 4},
    {"_tinytorch_C_torch_cross_entropy_loss", (DL_FUNC) &_tinytorch_C_torch_cross_entropy_loss, 6},
    {"_tinytorch_C_torch_triangular_solve", (DL_FUNC) &_tinytorch_C_torch_triangular_solve, 5},
    {"_tinytorch_C_torch_linalg_solve_triangular", (DL_FUNC) &_tinytorch_C_torch_linalg_solve_triangular, 5},
    {"_tinytorch_C_torch_linalg_vander", (DL_FUNC) &_tinytorch_C_torch_linalg_vander, 2},
    {"_tinytorch_C_torch_svd", (DL_FUNC) &_tinytorch_C_torch_svd, 3},
    {"_tinytorch_C_torch_swapaxes", (DL_FUNC) &_tinytorch_C_torch_swapaxes, 3},
    {"_tinytorch_C_torch_swapaxes_", (DL_FUNC) &_tinytorch_C_torch_swapaxes_, 3},
    {"_tinytorch_C_torch_swapdims", (DL_FUNC) &_tinytorch_C_torch_swapdims, 3},
    {"_tinytorch_C_torch_swapdims_", (DL_FUNC) &_tinytorch_C_torch_swapdims_, 3},
    {"_tinytorch_C_torch_cholesky", (DL_FUNC) &_tinytorch_C_torch_cholesky, 2},
    {"_tinytorch_C_torch_cholesky_solve", (DL_FUNC) &_tinytorch_C_torch_cholesky_solve, 3},
    {"_tinytorch_C_torch_cholesky_inverse", (DL_FUNC) &_tinytorch_C_torch_cholesky_inverse, 2},
    {"_tinytorch_C_torch_qr", (DL_FUNC) &_tinytorch_C_torch_qr, 2},
    {"_tinytorch_C_torch_geqrf", (DL_FUNC) &_tinytorch_C_torch_geqrf, 1},
    {"_tinytorch_C_torch_orgqr", (DL_FUNC) &_tinytorch_C_torch_orgqr, 2},
    {"_tinytorch_C_torch_ormqr", (DL_FUNC) &_tinytorch_C_torch_ormqr, 5},
    {"_tinytorch_C_torch_lu_solve", (DL_FUNC) &_tinytorch_C_torch_lu_solve, 3},
    {"_tinytorch_C_torch_lu_unpack", (DL_FUNC) &_tinytorch_C_torch_lu_unpack, 4},
    {"_tinytorch_C_torch_lgamma_", (DL_FUNC) &_tinytorch_C_torch_lgamma_, 1},
    {"_tinytorch_C_torch_lgamma", (DL_FUNC) &_tinytorch_C_torch_lgamma, 1},
    {"_tinytorch_C_torch_digamma", (DL_FUNC) &_tinytorch_C_torch_digamma, 1},
    {"_tinytorch_C_torch_polygamma", (DL_FUNC) &_tinytorch_C_torch_polygamma, 2},
    {"_tinytorch_C_torch_polygamma_", (DL_FUNC) &_tinytorch_C_torch_polygamma_, 2},
    {"_tinytorch_C_torch_erfinv", (DL_FUNC) &_tinytorch_C_torch_erfinv, 1},
    {"_tinytorch_C_torch_erfinv_", (DL_FUNC) &_tinytorch_C_torch_erfinv_, 1},
    {"_tinytorch_C_torch_i0", (DL_FUNC) &_tinytorch_C_torch_i0, 1},
    {"_tinytorch_C_torch_i0_", (DL_FUNC) &_tinytorch_C_torch_i0_, 1},
    {"_tinytorch_C_torch_sign_", (DL_FUNC) &_tinytorch_C_torch_sign_, 1},
    {"_tinytorch_C_torch_signbit", (DL_FUNC) &_tinytorch_C_torch_signbit, 1},
    {"_tinytorch_C_torch_dist", (DL_FUNC) &_tinytorch_C_torch_dist, 3},
    {"_tinytorch_C_torch_atan2_", (DL_FUNC) &_tinytorch_C_torch_atan2_, 2},
    {"_tinytorch_C_torch_atan2", (DL_FUNC) &_tinytorch_C_torch_atan2, 2},
    {"_tinytorch_C_torch_arctan2", (DL_FUNC) &_tinytorch_C_torch_arctan2, 2},
    {"_tinytorch_C_torch_arctan2_", (DL_FUNC) &_tinytorch_C_torch_arctan2_, 2},
    {"_tinytorch_C_torch_lerp", (DL_FUNC) &_tinytorch_C_torch_lerp, 3},
    {"_tinytorch_C_torch_histc", (DL_FUNC) &_tinytorch_C_torch_histc, 4},
    {"_tinytorch_C_torch_histogram", (DL_FUNC) &_tinytorch_C_torch_histogram, 4},
    {"_tinytorch_C_torch_histogramdd", (DL_FUNC) &_tinytorch_C_torch_histogramdd, 5},
    {"_tinytorch_C_torch_fmod", (DL_FUNC) &_tinytorch_C_torch_fmod, 2},
    {"_tinytorch_C_torch_fmod_", (DL_FUNC) &_tinytorch_C_torch_fmod_, 2},
    {"_tinytorch_C_torch_hypot", (DL_FUNC) &_tinytorch_C_torch_hypot, 2},
    {"_tinytorch_C_torch_hypot_", (DL_FUNC) &_tinytorch_C_torch_hypot_, 2},
    {"_tinytorch_C_torch_igamma", (DL_FUNC) &_tinytorch_C_torch_igamma, 2},
    {"_tinytorch_C_torch_igamma_", (DL_FUNC) &_tinytorch_C_torch_igamma_, 2},
    {"_tinytorch_C_torch_igammac", (DL_FUNC) &_tinytorch_C_torch_igammac, 2},
    {"_tinytorch_C_torch_igammac_", (DL_FUNC) &_tinytorch_C_torch_igammac_, 2},
    {"_tinytorch_C_torch_nextafter", (DL_FUNC) &_tinytorch_C_torch_nextafter, 2},
    {"_tinytorch_C_torch_nextafter_", (DL_FUNC) &_tinytorch_C_torch_nextafter_, 2},
    {"_tinytorch_C_torch_remainder_", (DL_FUNC) &_tinytorch_C_torch_remainder_, 2},
    {"_tinytorch_C_torch_fmin", (DL_FUNC) &_tinytorch_C_torch_fmin, 2},
    {"_tinytorch_C_torch_fmax", (DL_FUNC) &_tinytorch_C_torch_fmax, 2},
    {"_tinytorch_C_torch_minimum", (DL_FUNC) &_tinytorch_C_torch_minimum, 2},
    {"_tinytorch_C_torch_quantile", (DL_FUNC) &_tinytorch_C_torch_quantile, 5},
    {"_tinytorch_C_torch_nanquantile", (DL_FUNC) &_tinytorch_C_torch_nanquantile, 5},
    {"_tinytorch_C_torch_msort", (DL_FUNC) &_tinytorch_C_torch_msort, 1},
    {"_tinytorch_C_torch_argsort", (DL_FUNC) &_tinytorch_C_torch_argsort, 3},
    {"_tinytorch_C_torch_topk", (DL_FUNC) &_tinytorch_C_torch_topk, 5},
    {"_tinytorch_C_torch_renorm", (DL_FUNC) &_tinytorch_C_torch_renorm, 4},
    {"_tinytorch_C_torch_renorm_", (DL_FUNC) &_tinytorch_C_torch_renorm_, 4},
    {"_tinytorch_C_torch_unfold", (DL_FUNC) &_tinytorch_C_torch_unfold, 4},
    {"_tinytorch_C_torch_equal", (DL_FUNC) &_tinytorch_C_torch_equal, 2},
    {"_tinytorch_C_torch_pow_", (DL_FUNC) &_tinytorch_C_torch_pow_, 2},
    {"_tinytorch_C_torch_float_power", (DL_FUNC) &_tinytorch_C_torch_float_power, 2},
    {"_tinytorch_C_torch_float_power_", (DL_FUNC) &_tinytorch_C_torch_float_power_, 2},
    {"_tinytorch_C_torch_normal_functional", (DL_FUNC) &_tinytorch_C_torch_normal_functional, 4},
    {"_tinytorch_C_torch_normal", (DL_FUNC) &_tinytorch_C_torch_normal, 3},
    {"_tinytorch_C_torch_alias", (DL_FUNC) &_tinytorch_C_torch_alias, 1},
    {"_tinytorch_C_torch_bucketize", (DL_FUNC) &_tinytorch_C_torch_bucketize, 4},
    {"_tinytorch_C_torch_searchsorted", (DL_FUNC) &_tinytorch_C_torch_searchsorted, 6},
    {"_tinytorch_C_torch_mse_loss", (DL_FUNC) &_tinytorch_C_torch_mse_loss, 3},
    {"_tinytorch_C_torch_l1_loss", (DL_FUNC) &_tinytorch_C_torch_l1_loss, 3},
    {"_tinytorch_C_torch_multi_margin_loss", (DL_FUNC) &_tinytorch_C_torch_multi_margin_loss, 6},
    {"_tinytorch_C_torch_multilabel_margin_loss", (DL_FUNC) &_tinytorch_C_torch_multilabel_margin_loss, 3},
    {"_tinytorch_C_torch_multilabel_margin_loss_forward", (DL_FUNC) &_tinytorch_C_torch_multilabel_margin_loss_forward, 3},
    {"_tinytorch_C_torch_nll_loss_nd", (DL_FUNC) &_tinytorch_C_torch_nll_loss_nd, 5},
    {"_tinytorch_C_torch_nll_loss", (DL_FUNC) &_tinytorch_C_torch_nll_loss, 5},
    {"_tinytorch_C_torch_nll_loss_forward", (DL_FUNC) &_tinytorch_C_torch_nll_loss_forward, 5},
    {"_tinytorch_C_torch_nll_loss2d", (DL_FUNC) &_tinytorch_C_torch_nll_loss2d, 5},
    {"_tinytorch_C_torch_nll_loss2d_forward", (DL_FUNC) &_tinytorch_C_torch_nll_loss2d_forward, 5},
    {"_tinytorch_C_torch_smooth_l1_loss", (DL_FUNC) &_tinytorch_C_torch_smooth_l1_loss, 4},
    {"_tinytorch_C_torch_huber_loss", (DL_FUNC) &_tinytorch_C_torch_huber_loss, 4},
    {"_tinytorch_C_torch_soft_margin_loss", (DL_FUNC) &_tinytorch_C_torch_soft_margin_loss, 3},
    {"_tinytorch_C_torch_elu", (DL_FUNC) &_tinytorch_C_torch_elu, 4},
    {"_tinytorch_C_torch_elu_", (DL_FUNC) &_tinytorch_C_torch_elu_, 4},
    {"_tinytorch_C_torch_glu", (DL_FUNC) &_tinytorch_C_torch_glu, 2},
    {"_tinytorch_C_torch_glu_jvp", (DL_FUNC) &_tinytorch_C_torch_glu_jvp, 4},
    {"_tinytorch_C_torch_hardsigmoid", (DL_FUNC) &_tinytorch_C_torch_hardsigmoid, 1},
    {"_tinytorch_C_torch_hardsigmoid_", (DL_FUNC) &_tinytorch_C_torch_hardsigmoid_, 1},
    {"_tinytorch_C_torch_hardtanh", (DL_FUNC) &_tinytorch_C_torch_hardtanh, 3},
    {"_tinytorch_C_torch_hardtanh_", (DL_FUNC) &_tinytorch_C_torch_hardtanh_, 3},
    {"_tinytorch_C_torch_hardswish", (DL_FUNC) &_tinytorch_C_torch_hardswish, 1},
    {"_tinytorch_C_torch_hardswish_", (DL_FUNC) &_tinytorch_C_torch_hardswish_, 1},
    {"_tinytorch_C_torch_leaky_relu", (DL_FUNC) &_tinytorch_C_torch_leaky_relu, 2},
    {"_tinytorch_C_torch_leaky_relu_", (DL_FUNC) &_tinytorch_C_torch_leaky_relu_, 2},
    {"_tinytorch_C_torch_log_sigmoid", (DL_FUNC) &_tinytorch_C_torch_log_sigmoid, 1},
    {"_tinytorch_C_torch_log_sigmoid_forward", (DL_FUNC) &_tinytorch_C_torch_log_sigmoid_forward, 1},
    {"_tinytorch_C_torch_rrelu_with_noise", (DL_FUNC) &_tinytorch_C_torch_rrelu_with_noise, 6},
    {"_tinytorch_C_torch_rrelu_with_noise_", (DL_FUNC) &_tinytorch_C_torch_rrelu_with_noise_, 6},
    {"_tinytorch_C_torch_softplus", (DL_FUNC) &_tinytorch_C_torch_softplus, 3},
    {"_tinytorch_C_torch_softshrink", (DL_FUNC) &_tinytorch_C_torch_softshrink, 2},
    {"_tinytorch_C_torch_adaptive_avg_pool2d", (DL_FUNC) &_tinytorch_C_torch_adaptive_avg_pool2d, 2},
    {"_tinytorch_C_torch_adaptive_avg_pool3d", (DL_FUNC) &_tinytorch_C_torch_adaptive_avg_pool3d, 2},
    {"_tinytorch_C_torch_adaptive_max_pool2d", (DL_FUNC) &_tinytorch_C_torch_adaptive_max_pool2d, 2},
    {"_tinytorch_C_torch_adaptive_max_pool3d", (DL_FUNC) &_tinytorch_C_torch_adaptive_max_pool3d, 2},
    {"_tinytorch_C_torch_avg_pool2d", (DL_FUNC) &_tinytorch_C_torch_avg_pool2d, 7},
    {"_tinytorch_C_torch_avg_pool3d", (DL_FUNC) &_tinytorch_C_torch_avg_pool3d, 7},
    {"_tinytorch_C_torch_fractional_max_pool2d", (DL_FUNC) &_tinytorch_C_torch_fractional_max_pool2d, 4},
    {"_tinytorch_C_torch_fractional_max_pool3d", (DL_FUNC) &_tinytorch_C_torch_fractional_max_pool3d, 4},
    {"_tinytorch_C_torch_max_pool2d_with_indices", (DL_FUNC) &_tinytorch_C_torch_max_pool2d_with_indices, 6},
    {"_tinytorch_C_torch_max_pool3d_with_indices", (DL_FUNC) &_tinytorch_C_torch_max_pool3d_with_indices, 6},
    {"_tinytorch_C_torch_max_unpool2d", (DL_FUNC) &_tinytorch_C_torch_max_unpool2d, 3},
    {"_tinytorch_C_torch_max_unpool3d", (DL_FUNC) &_tinytorch_C_torch_max_unpool3d, 5},
    {"_tinytorch_C_torch_reflection_pad1d", (DL_FUNC) &_tinytorch_C_torch_reflection_pad1d, 2},
    {"_tinytorch_C_torch_reflection_pad2d", (DL_FUNC) &_tinytorch_C_torch_reflection_pad2d, 2},
    {"_tinytorch_C_torch_reflection_pad3d", (DL_FUNC) &_tinytorch_C_torch_reflection_pad3d, 2},
    {"_tinytorch_C_torch_replication_pad1d", (DL_FUNC) &_tinytorch_C_torch_replication_pad1d, 2},
    {"_tinytorch_C_torch_replication_pad2d", (DL_FUNC) &_tinytorch_C_torch_replication_pad2d, 2},
    {"_tinytorch_C_torch_replication_pad3d", (DL_FUNC) &_tinytorch_C_torch_replication_pad3d, 2},
    {"_tinytorch_C_torch_pad", (DL_FUNC) &_tinytorch_C_torch_pad, 4},
    {"_tinytorch_C_torch_upsample_linear1d", (DL_FUNC) &_tinytorch_C_torch_upsample_linear1d, 4},
    {"_tinytorch_C_torch_upsample_bilinear2d", (DL_FUNC) &_tinytorch_C_torch_upsample_bilinear2d, 4},
    {"_tinytorch_C_torch_upsample_trilinear3d", (DL_FUNC) &_tinytorch_C_torch_upsample_trilinear3d, 4},
    {"_tinytorch_C_torch_upsample_bicubic2d", (DL_FUNC) &_tinytorch_C_torch_upsample_bicubic2d, 4},
    {"_tinytorch_C_torch_upsample_nearest1d", (DL_FUNC) &_tinytorch_C_torch_upsample_nearest1d, 3},
    {"_tinytorch_C_torch_upsample_nearest2d", (DL_FUNC) &_tinytorch_C_torch_upsample_nearest2d, 3},
    {"_tinytorch_C_torch_upsample_nearest3d", (DL_FUNC) &_tinytorch_C_torch_upsample_nearest3d, 3},
    {"_tinytorch_C_torch_slow_conv_transpose2d", (DL_FUNC) &_tinytorch_C_torch_slow_conv_transpose2d, 8},
    {"_tinytorch_C_torch_slow_conv_transpose3d", (DL_FUNC) &_tinytorch_C_torch_slow_conv_transpose3d, 8},
    {"_tinytorch_C_torch_thnn_conv2d", (DL_FUNC) &_tinytorch_C_torch_thnn_conv2d, 6},
    {"_tinytorch_C_torch_conv_depthwise3d", (DL_FUNC) &_tinytorch_C_torch_conv_depthwise3d, 7},
    {"_tinytorch_C_torch_slow_conv3d", (DL_FUNC) &_tinytorch_C_torch_slow_conv3d, 6},
    {"_tinytorch_C_torch_slow_conv3d_forward", (DL_FUNC) &_tinytorch_C_torch_slow_conv3d_forward, 6},
    {"_tinytorch_C_torch_slow_conv_dilated2d", (DL_FUNC) &_tinytorch_C_torch_slow_conv_dilated2d, 7},
    {"_tinytorch_C_torch_slow_conv_dilated3d", (DL_FUNC) &_tinytorch_C_torch_slow_conv_dilated3d, 7},
    {"_tinytorch_C_torch_col2im", (DL_FUNC) &_tinytorch_C_torch_col2im, 6},
    {"_tinytorch_C_torch_column_stack", (DL_FUNC) &_tinytorch_C_torch_column_stack, 1},
    {"_tinytorch_C_torch_im2col", (DL_FUNC) &_tinytorch_C_torch_im2col, 5},
    {"_tinytorch_C_torch_isfinite", (DL_FUNC) &_tinytorch_C_torch_isfinite, 1},
    {"_tinytorch_C_torch_isinf", (DL_FUNC) &_tinytorch_C_torch_isinf, 1},
    {"_tinytorch_C_torch_isposinf", (DL_FUNC) &_tinytorch_C_torch_isposinf, 1},
    {"_tinytorch_C_torch_isneginf", (DL_FUNC) &_tinytorch_C_torch_isneginf, 1},
    {"_tinytorch_C_torch_special_entr", (DL_FUNC) &_tinytorch_C_torch_special_entr, 1},
    {"_tinytorch_C_torch_special_ndtri", (DL_FUNC) &_tinytorch_C_torch_special_ndtri, 1},
    {"_tinytorch_C_torch_special_log_ndtr", (DL_FUNC) &_tinytorch_C_torch_special_log_ndtr, 1},
    {"_tinytorch_C_torch_special_expm1", (DL_FUNC) &_tinytorch_C_torch_special_expm1, 1},
    {"_tinytorch_C_torch_special_exp2", (DL_FUNC) &_tinytorch_C_torch_special_exp2, 1},
    {"_tinytorch_C_torch_special_psi", (DL_FUNC) &_tinytorch_C_torch_special_psi, 1},
    {"_tinytorch_C_torch_special_digamma", (DL_FUNC) &_tinytorch_C_torch_special_digamma, 1},
    {"_tinytorch_C_torch_special_gammaln", (DL_FUNC) &_tinytorch_C_torch_special_gammaln, 1},
    {"_tinytorch_C_torch_special_erf", (DL_FUNC) &_tinytorch_C_torch_special_erf, 1},
    {"_tinytorch_C_torch_special_erfc", (DL_FUNC) &_tinytorch_C_torch_special_erfc, 1},
    {"_tinytorch_C_torch_special_erfcx", (DL_FUNC) &_tinytorch_C_torch_special_erfcx, 1},
    {"_tinytorch_C_torch_special_erfinv", (DL_FUNC) &_tinytorch_C_torch_special_erfinv, 1},
    {"_tinytorch_C_torch_special_ndtr", (DL_FUNC) &_tinytorch_C_torch_special_ndtr, 1},
    {"_tinytorch_C_torch_special_xlog1py", (DL_FUNC) &_tinytorch_C_torch_special_xlog1py, 2},
    {"_tinytorch_C_torch_special_xlogy", (DL_FUNC) &_tinytorch_C_torch_special_xlogy, 2},
    {"_tinytorch_C_torch_special_zeta", (DL_FUNC) &_tinytorch_C_torch_special_zeta, 2},
    {"_tinytorch_C_torch_special_i0", (DL_FUNC) &_tinytorch_C_torch_special_i0, 1},
    {"_tinytorch_C_torch_special_i0e", (DL_FUNC) &_tinytorch_C_torch_special_i0e, 1},
    {"_tinytorch_C_torch_special_i1", (DL_FUNC) &_tinytorch_C_torch_special_i1, 1},
    {"_tinytorch_C_torch_special_i1e", (DL_FUNC) &_tinytorch_C_torch_special_i1e, 1},
    {"_tinytorch_C_torch_special_logit", (DL_FUNC) &_tinytorch_C_torch_special_logit, 2},
    {"_tinytorch_C_torch_special_polygamma", (DL_FUNC) &_tinytorch_C_torch_special_polygamma, 2},
    {"_tinytorch_C_torch_special_logsumexp", (DL_FUNC) &_tinytorch_C_torch_special_logsumexp, 3},
    {"_tinytorch_C_torch_special_expit", (DL_FUNC) &_tinytorch_C_torch_special_expit, 1},
    {"_tinytorch_C_torch_special_sinc", (DL_FUNC) &_tinytorch_C_torch_special_sinc, 1},
    {"_tinytorch_C_torch_special_round", (DL_FUNC) &_tinytorch_C_torch_special_round, 2},
    {"_tinytorch_C_torch_special_log1p", (DL_FUNC) &_tinytorch_C_torch_special_log1p, 1},
    {"_tinytorch_C_torch_special_log_softmax", (DL_FUNC) &_tinytorch_C_torch_special_log_softmax, 3},
    {"_tinytorch_C_torch_special_gammainc", (DL_FUNC) &_tinytorch_C_torch_special_gammainc, 2},
    {"_tinytorch_C_torch_special_gammaincc", (DL_FUNC) &_tinytorch_C_torch_special_gammaincc, 2},
    {"_tinytorch_C_torch_special_multigammaln", (DL_FUNC) &_tinytorch_C_torch_special_multigammaln, 2},
    {"_tinytorch_C_torch_special_softmax", (DL_FUNC) &_tinytorch_C_torch_special_softmax, 3},
    {"_tinytorch_C_torch_fft_fft", (DL_FUNC) &_tinytorch_C_torch_fft_fft, 4},
    {"_tinytorch_C_torch_fft_ifft", (DL_FUNC) &_tinytorch_C_torch_fft_ifft, 4},
    {"_tinytorch_C_torch_fft_rfft", (DL_FUNC) &_tinytorch_C_torch_fft_rfft, 4},
    {"_tinytorch_C_torch_fft_irfft", (DL_FUNC) &_tinytorch_C_torch_fft_irfft, 4},
    {"_tinytorch_C_torch_fft_hfft", (DL_FUNC) &_tinytorch_C_torch_fft_hfft, 4},
    {"_tinytorch_C_torch_fft_ihfft", (DL_FUNC) &_tinytorch_C_torch_fft_ihfft, 4},
    {"_tinytorch_C_torch_fft_fft2", (DL_FUNC) &_tinytorch_C_torch_fft_fft2, 4},
    {"_tinytorch_C_torch_fft_ifft2", (DL_FUNC) &_tinytorch_C_torch_fft_ifft2, 4},
    {"_tinytorch_C_torch_fft_rfft2", (DL_FUNC) &_tinytorch_C_torch_fft_rfft2, 4},
    {"_tinytorch_C_torch_fft_irfft2", (DL_FUNC) &_tinytorch_C_torch_fft_irfft2, 4},
    {"_tinytorch_C_torch_fft_hfft2", (DL_FUNC) &_tinytorch_C_torch_fft_hfft2, 4},
    {"_tinytorch_C_torch_fft_ihfft2", (DL_FUNC) &_tinytorch_C_torch_fft_ihfft2, 4},
    {"_tinytorch_C_torch_fft_fftn", (DL_FUNC) &_tinytorch_C_torch_fft_fftn, 4},
    {"_tinytorch_C_torch_fft_ifftn", (DL_FUNC) &_tinytorch_C_torch_fft_ifftn, 4},
    {"_tinytorch_C_torch_fft_rfftn", (DL_FUNC) &_tinytorch_C_torch_fft_rfftn, 4},
    {"_tinytorch_C_torch_fft_irfftn", (DL_FUNC) &_tinytorch_C_torch_fft_irfftn, 4},
    {"_tinytorch_C_torch_fft_hfftn", (DL_FUNC) &_tinytorch_C_torch_fft_hfftn, 4},
    {"_tinytorch_C_torch_fft_ihfftn", (DL_FUNC) &_tinytorch_C_torch_fft_ihfftn, 4},
    {"_tinytorch_C_torch_fft_fftfreq", (DL_FUNC) &_tinytorch_C_torch_fft_fftfreq, 4},
    {"_tinytorch_C_torch_fft_rfftfreq", (DL_FUNC) &_tinytorch_C_torch_fft_rfftfreq, 4},
    {"_tinytorch_C_torch_fft_fftshift", (DL_FUNC) &_tinytorch_C_torch_fft_fftshift, 2},
    {"_tinytorch_C_torch_fft_ifftshift", (DL_FUNC) &_tinytorch_C_torch_fft_ifftshift, 2},
    {"_tinytorch_C_torch_linalg_cholesky_ex", (DL_FUNC) &_tinytorch_C_torch_linalg_cholesky_ex, 3},
    {"_tinytorch_C_torch_linalg_cholesky", (DL_FUNC) &_tinytorch_C_torch_linalg_cholesky, 2},
    {"_tinytorch_C_torch_linalg_cross", (DL_FUNC) &_tinytorch_C_torch_linalg_cross, 3},
    {"_tinytorch_C_torch_linalg_lu_factor", (DL_FUNC) &_tinytorch_C_torch_linalg_lu_factor, 2},
    {"_tinytorch_C_torch_linalg_lu_factor_ex", (DL_FUNC) &_tinytorch_C_torch_linalg_lu_factor_ex, 3},
    {"_tinytorch_C_torch_linalg_lu", (DL_FUNC) &_tinytorch_C_torch_linalg_lu, 2},
    {"_tinytorch_C_torch_linalg_lu_solve", (DL_FUNC) &_tinytorch_C_torch_linalg_lu_solve, 5},
    {"_tinytorch_C_torch_linalg_det", (DL_FUNC) &_tinytorch_C_torch_linalg_det, 1},
    {"_tinytorch_C_torch_det", (DL_FUNC) &_tinytorch_C_torch_det, 1},
    {"_tinytorch_C_torch_linalg_ldl_factor_ex", (DL_FUNC) &_tinytorch_C_torch_linalg_ldl_factor_ex, 3},
    {"_tinytorch_C_torch_linalg_ldl_factor", (DL_FUNC) &_tinytorch_C_torch_linalg_ldl_factor, 2},
    {"_tinytorch_C_torch_linalg_ldl_solve", (DL_FUNC) &_tinytorch_C_torch_linalg_ldl_solve, 4},
    {"_tinytorch_C_torch_linalg_lstsq", (DL_FUNC) &_tinytorch_C_torch_linalg_lstsq, 4},
    {"_tinytorch_C_torch_linalg_matmul", (DL_FUNC) &_tinytorch_C_torch_linalg_matmul, 2},
    {"_tinytorch_C_torch_linalg_vecdot", (DL_FUNC) &_tinytorch_C_torch_linalg_vecdot, 3},
    {"_tinytorch_C_torch_linalg_matrix_exp", (DL_FUNC) &_tinytorch_C_torch_linalg_matrix_exp, 1},
    {"_tinytorch_C_torch_linalg_slogdet", (DL_FUNC) &_tinytorch_C_torch_linalg_slogdet, 1},
    {"_tinytorch_C_torch_slogdet", (DL_FUNC) &_tinytorch_C_torch_slogdet, 1},
    {"_tinytorch_C_torch_logdet", (DL_FUNC) &_tinytorch_C_torch_logdet, 1},
    {"_tinytorch_C_torch_linalg_eig", (DL_FUNC) &_tinytorch_C_torch_linalg_eig, 1},
    {"_tinytorch_C_torch_linalg_eigvals", (DL_FUNC) &_tinytorch_C_torch_linalg_eigvals, 1},
    {"_tinytorch_C_torch_linalg_eigh", (DL_FUNC) &_tinytorch_C_torch_linalg_eigh, 2},
    {"_tinytorch_C_torch_linalg_eigvalsh", (DL_FUNC) &_tinytorch_C_torch_linalg_eigvalsh, 2},
    {"_tinytorch_C_torch_linalg_householder_product", (DL_FUNC) &_tinytorch_C_torch_linalg_householder_product, 2},
    {"_tinytorch_C_torch_linalg_inv_ex", (DL_FUNC) &_tinytorch_C_torch_linalg_inv_ex, 2},
    {"_tinytorch_C_torch_linalg_inv", (DL_FUNC) &_tinytorch_C_torch_linalg_inv, 1},
    {"_tinytorch_C_torch_inverse", (DL_FUNC) &_tinytorch_C_torch_inverse, 1},
    {"_tinytorch_C_torch_inner", (DL_FUNC) &_tinytorch_C_torch_inner, 2},
    {"_tinytorch_C_torch_ger", (DL_FUNC) &_tinytorch_C_torch_ger, 2},
    {"_tinytorch_C_torch_linalg_norm", (DL_FUNC) &_tinytorch_C_torch_linalg_norm, 5},
    {"_tinytorch_C_torch_linalg_vector_norm", (DL_FUNC) &_tinytorch_C_torch_linalg_vector_norm, 5},
    {"_tinytorch_C_torch_linalg_matrix_norm", (DL_FUNC) &_tinytorch_C_torch_linalg_matrix_norm, 5},
    {"_tinytorch_C_torch_linalg_svd", (DL_FUNC) &_tinytorch_C_torch_linalg_svd, 3},
    {"_tinytorch_C_torch_linalg_svdvals", (DL_FUNC) &_tinytorch_C_torch_linalg_svdvals, 2},
    {"_tinytorch_C_torch_linalg_cond", (DL_FUNC) &_tinytorch_C_torch_linalg_cond, 2},
    {"_tinytorch_C_torch_linalg_pinv", (DL_FUNC) &_tinytorch_C_torch_linalg_pinv, 4},
    {"_tinytorch_C_torch_linalg_solve_ex", (DL_FUNC) &_tinytorch_C_torch_linalg_solve_ex, 4},
    {"_tinytorch_C_torch_linalg_solve", (DL_FUNC) &_tinytorch_C_torch_linalg_solve, 3},
    {"_tinytorch_C_torch_linalg_tensorinv", (DL_FUNC) &_tinytorch_C_torch_linalg_tensorinv, 2},
    {"_tinytorch_C_torch_linalg_tensorsolve", (DL_FUNC) &_tinytorch_C_torch_linalg_tensorsolve, 3},
    {"_tinytorch_C_torch_linalg_qr", (DL_FUNC) &_tinytorch_C_torch_linalg_qr, 2},
    {"_tinytorch_C_torch_linalg_matrix_power", (DL_FUNC) &_tinytorch_C_torch_linalg_matrix_power, 2},
    {"_tinytorch_C_torch_linalg_matrix_rank", (DL_FUNC) &_tinytorch_C_torch_linalg_matrix_rank, 4},
    {"_tinytorch_C_torch_linalg_multi_dot", (DL_FUNC) &_tinytorch_C_torch_linalg_multi_dot, 1},
    {"_tinytorch_C_torch_nested_to_padded_tensor", (DL_FUNC) &_tinytorch_C_torch_nested_to_padded_tensor, 3},
    {"_tinytorch_C_torch_segment_reduce", (DL_FUNC) &_tinytorch_C_torch_segment_reduce, 8},
    {"_tinytorch_C_torch_pad_sequence", (DL_FUNC) &_tinytorch_C_torch_pad_sequence, 4},
    {"_tinytorch_C_torch_flatten_dense_tensors", (DL_FUNC) &_tinytorch_C_torch_flatten_dense_tensors, 1},
    {"_tinytorch_C_torch_unflatten_dense_tensors", (DL_FUNC) &_tinytorch_C_torch_unflatten_dense_tensors, 2},
    {"_tinytorch_C_torch_to_padded_tensor", (DL_FUNC) &_tinytorch_C_torch_to_padded_tensor, 3},
    {"_tinytorch_C_torch_special_airy_ai", (DL_FUNC) &_tinytorch_C_torch_special_airy_ai, 1},
    {"_tinytorch_C_torch_special_bessel_j0", (DL_FUNC) &_tinytorch_C_torch_special_bessel_j0, 1},
    {"_tinytorch_C_torch_special_bessel_j1", (DL_FUNC) &_tinytorch_C_torch_special_bessel_j1, 1},
    {"_tinytorch_C_torch_special_bessel_y0", (DL_FUNC) &_tinytorch_C_torch_special_bessel_y0, 1},
    {"_tinytorch_C_torch_special_bessel_y1", (DL_FUNC) &_tinytorch_C_torch_special_bessel_y1, 1},
    {"_tinytorch_C_torch_special_chebyshev_polynomial_t", (DL_FUNC) &_tinytorch_C_torch_special_chebyshev_polynomial_t, 2},
    {"_tinytorch_C_torch_special_chebyshev_polynomial_u", (DL_FUNC) &_tinytorch_C_torch_special_chebyshev_polynomial_u, 2},
    {"_tinytorch_C_torch_special_chebyshev_polynomial_v", (DL_FUNC) &_tinytorch_C_torch_special_chebyshev_polynomial_v, 2},
    {"_tinytorch_C_torch_special_chebyshev_polynomial_w", (DL_FUNC) &_tinytorch_C_torch_special_chebyshev_polynomial_w, 2},
    {"_tinytorch_C_torch_special_hermite_polynomial_h", (DL_FUNC) &_tinytorch_C_torch_special_hermite_polynomial_h, 2},
    {"_tinytorch_C_torch_special_hermite_polynomial_he", (DL_FUNC) &_tinytorch_C_torch_special_hermite_polynomial_he, 2},
    {"_tinytorch_C_torch_special_laguerre_polynomial_l", (DL_FUNC) &_tinytorch_C_torch_special_laguerre_polynomial_l, 2},
    {"_tinytorch_C_torch_special_legendre_polynomial_p", (DL_FUNC) &_tinytorch_C_torch_special_legendre_polynomial_p, 2},
    {"_tinytorch_C_torch_special_modified_bessel_i0", (DL_FUNC) &_tinytorch_C_torch_special_modified_bessel_i0, 1},
    {"_tinytorch_C_torch_special_modified_bessel_i1", (DL_FUNC) &_tinytorch_C_torch_special_modified_bessel_i1, 1},
    {"_tinytorch_C_torch_special_modified_bessel_k0", (DL_FUNC) &_tinytorch_C_torch_special_modified_bessel_k0, 1},
    {"_tinytorch_C_torch_special_modified_bessel_k1", (DL_FUNC) &_tinytorch_C_torch_special_modified_bessel_k1, 1},
    {"_tinytorch_C_torch_special_scaled_modified_bessel_k0", (DL_FUNC) &_tinytorch_C_torch_special_scaled_modified_bessel_k0, 1},
    {"_tinytorch_C_torch_special_scaled_modified_bessel_k1", (DL_FUNC) &_tinytorch_C_torch_special_scaled_modified_bessel_k1, 1},
    {"_tinytorch_C_torch_special_shifted_chebyshev_polynomial_t", (DL_FUNC) &_tinytorch_C_torch_special_shifted_chebyshev_polynomial_t, 2},
    {"_tinytorch_C_torch_special_shifted_chebyshev_polynomial_u", (DL_FUNC) &_tinytorch_C_torch_special_shifted_chebyshev_polynomial_u, 2},
    {"_tinytorch_C_torch_special_shifted_chebyshev_polynomial_v", (DL_FUNC) &_tinytorch_C_torch_special_shifted_chebyshev_polynomial_v, 2},
    {"_tinytorch_C_torch_special_shifted_chebyshev_polynomial_w", (DL_FUNC) &_tinytorch_C_torch_special_shifted_chebyshev_polynomial_w, 2},
    {"_tinytorch_C_torch_special_spherical_bessel_j0", (DL_FUNC) &_tinytorch_C_torch_special_spherical_bessel_j0, 1},
    {"_tinytorch_C_gpu_launch", (DL_FUNC) &_tinytorch_C_gpu_launch, 7},
    {"_tinytorch_C_gpu_launch_reduction", (DL_FUNC) &_tinytorch_C_gpu_launch_reduction, 8},
    {"_tinytorch_C_gpu_launch_generic", (DL_FUNC) &_tinytorch_C_gpu_launch_generic, 7},
    {"_tinytorch_C_gpu_kernel_cache_clear", (DL_FUNC) &_tinytorch_C_gpu_kernel_cache_clear, 0},
    {"_tinytorch_C_gpu_kernel_cache_stats", (DL_FUNC) &_tinytorch_C_gpu_kernel_cache_stats, 0},
    {"_tinytorch_C_torch_index", (DL_FUNC) &_tinytorch_C_torch_index, 3},
    {"_tinytorch_C_torch_index_put", (DL_FUNC) &_tinytorch_C_torch_index_put, 3},
    {"_tinytorch_C_nnf_silu", (DL_FUNC) &_tinytorch_C_nnf_silu, 1},
    {"_tinytorch_C_nnf_gelu", (DL_FUNC) &_tinytorch_C_nnf_gelu, 2},
    {"_tinytorch_C_nnf_leaky_relu", (DL_FUNC) &_tinytorch_C_nnf_leaky_relu, 2},
    {"_tinytorch_C_nnf_elu", (DL_FUNC) &_tinytorch_C_nnf_elu, 2},
    {"_tinytorch_C_nnf_softmax", (DL_FUNC) &_tinytorch_C_nnf_softmax, 2},
    {"_tinytorch_C_nnf_log_softmax", (DL_FUNC) &_tinytorch_C_nnf_log_softmax, 2},
    {"_tinytorch_C_nnf_layer_norm", (DL_FUNC) &_tinytorch_C_nnf_layer_norm, 5},
    {"_tinytorch_C_torch_linear", (DL_FUNC) &_tinytorch_C_torch_linear, 3},
    {"_tinytorch_C_torch_conv1d", (DL_FUNC) &_tinytorch_C_torch_conv1d, 7},
    {"_tinytorch_C_torch_embedding", (DL_FUNC) &_tinytorch_C_torch_embedding, 2},
    {"_tinytorch_C_torch_conv_transpose1d", (DL_FUNC) &_tinytorch_C_torch_conv_transpose1d, 8},
    {"_tinytorch_C_torch_conv2d", (DL_FUNC) &_tinytorch_C_torch_conv2d, 7},
    {"_tinytorch_C_torch_batch_norm", (DL_FUNC) &_tinytorch_C_torch_batch_norm, 9},
    {"_tinytorch_C_torch_lstm", (DL_FUNC) &_tinytorch_C_torch_lstm, 8},
    {"_tinytorch_C_nnf_pad", (DL_FUNC) &_tinytorch_C_nnf_pad, 4},
    {"_tinytorch_C_nnf_interpolate", (DL_FUNC) &_tinytorch_C_nnf_interpolate, 5},
    {"_tinytorch_C_nnf_avg_pool1d", (DL_FUNC) &_tinytorch_C_nnf_avg_pool1d, 6},
    {"_tinytorch_C_nnf_softplus", (DL_FUNC) &_tinytorch_C_nnf_softplus, 3},
    {"_tinytorch_C_nnf_normalize", (DL_FUNC) &_tinytorch_C_nnf_normalize, 4},
    {"_tinytorch_C_torch_sdpa", (DL_FUNC) &_tinytorch_C_torch_sdpa, 6},
    {"_tinytorch_C_torch_add", (DL_FUNC) &_tinytorch_C_torch_add, 3},
    {"_tinytorch_C_torch_sub", (DL_FUNC) &_tinytorch_C_torch_sub, 3},
    {"_tinytorch_C_torch_mul", (DL_FUNC) &_tinytorch_C_torch_mul, 2},
    {"_tinytorch_C_torch_div", (DL_FUNC) &_tinytorch_C_torch_div, 2},
    {"_tinytorch_C_torch_neg", (DL_FUNC) &_tinytorch_C_torch_neg, 1},
    {"_tinytorch_C_torch_logical_not", (DL_FUNC) &_tinytorch_C_torch_logical_not, 1},
    {"_tinytorch_C_torch_add_scalar", (DL_FUNC) &_tinytorch_C_torch_add_scalar, 2},
    {"_tinytorch_C_torch_sub_scalar", (DL_FUNC) &_tinytorch_C_torch_sub_scalar, 2},
    {"_tinytorch_C_torch_mul_scalar", (DL_FUNC) &_tinytorch_C_torch_mul_scalar, 2},
    {"_tinytorch_C_torch_div_scalar", (DL_FUNC) &_tinytorch_C_torch_div_scalar, 2},
    {"_tinytorch_C_torch_matmul", (DL_FUNC) &_tinytorch_C_torch_matmul, 2},
    {"_tinytorch_C_torch_mm", (DL_FUNC) &_tinytorch_C_torch_mm, 2},
    {"_tinytorch_C_torch_mm_dtype", (DL_FUNC) &_tinytorch_C_torch_mm_dtype, 3},
    {"_tinytorch_C_torch_t", (DL_FUNC) &_tinytorch_C_torch_t, 1},
    {"_tinytorch_C_torch_sum", (DL_FUNC) &_tinytorch_C_torch_sum, 3},
    {"_tinytorch_C_torch_mean", (DL_FUNC) &_tinytorch_C_torch_mean, 3},
    {"_tinytorch_C_torch_max", (DL_FUNC) &_tinytorch_C_torch_max, 2},
    {"_tinytorch_C_torch_min", (DL_FUNC) &_tinytorch_C_torch_min, 2},
    {"_tinytorch_C_torch_argmax", (DL_FUNC) &_tinytorch_C_torch_argmax, 3},
    {"_tinytorch_C_torch_argmin", (DL_FUNC) &_tinytorch_C_torch_argmin, 3},
    {"_tinytorch_C_torch_reshape", (DL_FUNC) &_tinytorch_C_torch_reshape, 2},
    {"_tinytorch_C_torch_view", (DL_FUNC) &_tinytorch_C_torch_view, 2},
    {"_tinytorch_C_torch_squeeze", (DL_FUNC) &_tinytorch_C_torch_squeeze, 2},
    {"_tinytorch_C_torch_unsqueeze", (DL_FUNC) &_tinytorch_C_torch_unsqueeze, 2},
    {"_tinytorch_C_torch_relu", (DL_FUNC) &_tinytorch_C_torch_relu, 1},
    {"_tinytorch_C_torch_sigmoid", (DL_FUNC) &_tinytorch_C_torch_sigmoid, 1},
    {"_tinytorch_C_torch_tanh", (DL_FUNC) &_tinytorch_C_torch_tanh, 1},
    {"_tinytorch_C_torch_exp", (DL_FUNC) &_tinytorch_C_torch_exp, 1},
    {"_tinytorch_C_torch_log", (DL_FUNC) &_tinytorch_C_torch_log, 1},
    {"_tinytorch_C_torch_log2", (DL_FUNC) &_tinytorch_C_torch_log2, 1},
    {"_tinytorch_C_torch_log10", (DL_FUNC) &_tinytorch_C_torch_log10, 1},
    {"_tinytorch_C_torch_sqrt", (DL_FUNC) &_tinytorch_C_torch_sqrt, 1},
    {"_tinytorch_C_torch_abs", (DL_FUNC) &_tinytorch_C_torch_abs, 1},
    {"_tinytorch_C_torch_sign", (DL_FUNC) &_tinytorch_C_torch_sign, 1},
    {"_tinytorch_C_torch_floor", (DL_FUNC) &_tinytorch_C_torch_floor, 1},
    {"_tinytorch_C_torch_ceil", (DL_FUNC) &_tinytorch_C_torch_ceil, 1},
    {"_tinytorch_C_torch_round", (DL_FUNC) &_tinytorch_C_torch_round, 1},
    {"_tinytorch_C_torch_trunc", (DL_FUNC) &_tinytorch_C_torch_trunc, 1},
    {"_tinytorch_C_torch_sin", (DL_FUNC) &_tinytorch_C_torch_sin, 1},
    {"_tinytorch_C_torch_cos", (DL_FUNC) &_tinytorch_C_torch_cos, 1},
    {"_tinytorch_C_torch_rsqrt", (DL_FUNC) &_tinytorch_C_torch_rsqrt, 1},
    {"_tinytorch_C_torch_detach", (DL_FUNC) &_tinytorch_C_torch_detach, 1},
    {"_tinytorch_C_torch_pow", (DL_FUNC) &_tinytorch_C_torch_pow, 2},
    {"_tinytorch_C_torch_pow_scalar", (DL_FUNC) &_tinytorch_C_torch_pow_scalar, 2},
    {"_tinytorch_C_torch_scalar_pow", (DL_FUNC) &_tinytorch_C_torch_scalar_pow, 2},
    {"_tinytorch_C_torch_remainder", (DL_FUNC) &_tinytorch_C_torch_remainder, 2},
    {"_tinytorch_C_torch_remainder_scalar", (DL_FUNC) &_tinytorch_C_torch_remainder_scalar, 2},
    {"_tinytorch_C_torch_floor_divide", (DL_FUNC) &_tinytorch_C_torch_floor_divide, 2},
    {"_tinytorch_C_torch_floor_divide_scalar", (DL_FUNC) &_tinytorch_C_torch_floor_divide_scalar, 2},
    {"_tinytorch_C_torch_eq", (DL_FUNC) &_tinytorch_C_torch_eq, 2},
    {"_tinytorch_C_torch_eq_scalar", (DL_FUNC) &_tinytorch_C_torch_eq_scalar, 2},
    {"_tinytorch_C_torch_ne", (DL_FUNC) &_tinytorch_C_torch_ne, 2},
    {"_tinytorch_C_torch_ne_scalar", (DL_FUNC) &_tinytorch_C_torch_ne_scalar, 2},
    {"_tinytorch_C_torch_lt", (DL_FUNC) &_tinytorch_C_torch_lt, 2},
    {"_tinytorch_C_torch_lt_scalar", (DL_FUNC) &_tinytorch_C_torch_lt_scalar, 2},
    {"_tinytorch_C_torch_le", (DL_FUNC) &_tinytorch_C_torch_le, 2},
    {"_tinytorch_C_torch_le_scalar", (DL_FUNC) &_tinytorch_C_torch_le_scalar, 2},
    {"_tinytorch_C_torch_gt", (DL_FUNC) &_tinytorch_C_torch_gt, 2},
    {"_tinytorch_C_torch_gt_scalar", (DL_FUNC) &_tinytorch_C_torch_gt_scalar, 2},
    {"_tinytorch_C_torch_ge", (DL_FUNC) &_tinytorch_C_torch_ge, 2},
    {"_tinytorch_C_torch_ge_scalar", (DL_FUNC) &_tinytorch_C_torch_ge_scalar, 2},
    {"_tinytorch_C_torch_bmm", (DL_FUNC) &_tinytorch_C_torch_bmm, 2},
    {"_tinytorch_C_torch_bmm_dtype", (DL_FUNC) &_tinytorch_C_torch_bmm_dtype, 3},
    {"_tinytorch_C_torch_transpose", (DL_FUNC) &_tinytorch_C_torch_transpose, 3},
    {"_tinytorch_C_torch_flatten", (DL_FUNC) &_tinytorch_C_torch_flatten, 3},
    {"_tinytorch_C_torch_clone", (DL_FUNC) &_tinytorch_C_torch_clone, 1},
    {"_tinytorch_C_torch_contiguous", (DL_FUNC) &_tinytorch_C_torch_contiguous, 1},
    {"_tinytorch_C_torch_to_dtype", (DL_FUNC) &_tinytorch_C_torch_to_dtype, 2},
    {"_tinytorch_C_torch_item", (DL_FUNC) &_tinytorch_C_torch_item, 1},
    {"_tinytorch_C_torch_cat", (DL_FUNC) &_tinytorch_C_torch_cat, 2},
    {"_tinytorch_C_torch_clamp", (DL_FUNC) &_tinytorch_C_torch_clamp, 3},
    {"_tinytorch_C_torch_where", (DL_FUNC) &_tinytorch_C_torch_where, 3},
    {"_tinytorch_C_torch_sort", (DL_FUNC) &_tinytorch_C_torch_sort, 3},
    {"_tinytorch_C_torch_flip", (DL_FUNC) &_tinytorch_C_torch_flip, 2},
    {"_tinytorch_C_torch_cumsum", (DL_FUNC) &_tinytorch_C_torch_cumsum, 2},
    {"_tinytorch_C_torch_maximum", (DL_FUNC) &_tinytorch_C_torch_maximum, 2},
    {"_tinytorch_C_torch_multinomial", (DL_FUNC) &_tinytorch_C_torch_multinomial, 3},
    {"_tinytorch_C_torch_outer", (DL_FUNC) &_tinytorch_C_torch_outer, 2},
    {"_tinytorch_C_torch_triu", (DL_FUNC) &_tinytorch_C_torch_triu, 2},
    {"_tinytorch_C_torch_norm", (DL_FUNC) &_tinytorch_C_torch_norm, 4},
    {"_tinytorch_C_torch_std", (DL_FUNC) &_tinytorch_C_torch_std, 4},
    {"_tinytorch_C_torch_complex", (DL_FUNC) &_tinytorch_C_torch_complex, 2},
    {"_tinytorch_C_torch_real", (DL_FUNC) &_tinytorch_C_torch_real, 1},
    {"_tinytorch_C_torch_imag", (DL_FUNC) &_tinytorch_C_torch_imag, 1},
    {"_tinytorch_C_torch_polar", (DL_FUNC) &_tinytorch_C_torch_polar, 2},
    {"_tinytorch_C_torch_view_as_real", (DL_FUNC) &_tinytorch_C_torch_view_as_real, 1},
    {"_tinytorch_C_torch_stft", (DL_FUNC) &_tinytorch_C_torch_stft, 9},
    {"_tinytorch_C_torch_istft", (DL_FUNC) &_tinytorch_C_torch_istft, 10},
    {"_tinytorch_C_torch_hann_window", (DL_FUNC) &_tinytorch_C_torch_hann_window, 4},
    {"_tinytorch_C_torch_addmm_dtype", (DL_FUNC) &_tinytorch_C_torch_addmm_dtype, 6},
    {"_tinytorch_C_torch_baddbmm_dtype", (DL_FUNC) &_tinytorch_C_torch_baddbmm_dtype, 6},
    {"_tinytorch_C_optim_sgd", (DL_FUNC) &_tinytorch_C_optim_sgd, 6},
    {"_tinytorch_C_optim_adam", (DL_FUNC) &_tinytorch_C_optim_adam, 7},
    {"_tinytorch_C_optim_adamw", (DL_FUNC) &_tinytorch_C_optim_adamw, 7},
    {"_tinytorch_C_optim_step", (DL_FUNC) &_tinytorch_C_optim_step, 1},
    {"_tinytorch_C_optim_zero_grad", (DL_FUNC) &_tinytorch_C_optim_zero_grad, 2},
    {"_tinytorch_C_rtorch_ping", (DL_FUNC) &_tinytorch_C_rtorch_ping, 0},
    {"_tinytorch_C_torch_tensor", (DL_FUNC) &_tinytorch_C_torch_tensor, 3},
    {"_tinytorch_C_torch_tensor_raw", (DL_FUNC) &_tinytorch_C_torch_tensor_raw, 3},
    {"_tinytorch_C_torch_zeros", (DL_FUNC) &_tinytorch_C_torch_zeros, 3},
    {"_tinytorch_C_torch_ones", (DL_FUNC) &_tinytorch_C_torch_ones, 3},
    {"_tinytorch_C_torch_randn", (DL_FUNC) &_tinytorch_C_torch_randn, 3},
    {"_tinytorch_C_torch_empty_like", (DL_FUNC) &_tinytorch_C_torch_empty_like, 1},
    {"_tinytorch_C_torch_empty", (DL_FUNC) &_tinytorch_C_torch_empty, 3},
    {"_tinytorch_C_torch_tensor_from_buffer", (DL_FUNC) &_tinytorch_C_torch_tensor_from_buffer, 4},
    {"_tinytorch_C_torch_arange", (DL_FUNC) &_tinytorch_C_torch_arange, 5},
    {"_tinytorch_C_torch_full", (DL_FUNC) &_tinytorch_C_torch_full, 4},
    {"_tinytorch_C_torch_linspace", (DL_FUNC) &_tinytorch_C_torch_linspace, 5},
    {"_tinytorch_C_torch_ones_like", (DL_FUNC) &_tinytorch_C_torch_ones_like, 2},
    {"_tinytorch_C_torch_zeros_like", (DL_FUNC) &_tinytorch_C_torch_zeros_like, 2},
    {"_tinytorch_C_torch_randn_like", (DL_FUNC) &_tinytorch_C_torch_randn_like, 2},
    {"_tinytorch_C_torch_permute", (DL_FUNC) &_tinytorch_C_torch_permute, 2},
    {"_tinytorch_C_torch_expand", (DL_FUNC) &_tinytorch_C_torch_expand, 2},
    {"_tinytorch_C_torch_gather", (DL_FUNC) &_tinytorch_C_torch_gather, 3},
    {"_tinytorch_C_torch_masked_fill", (DL_FUNC) &_tinytorch_C_torch_masked_fill, 3},
    {"_tinytorch_C_torch_masked_fill_", (DL_FUNC) &_tinytorch_C_torch_masked_fill_, 3},
    {"_tinytorch_C_torch_copy_", (DL_FUNC) &_tinytorch_C_torch_copy_, 2},
    {"_tinytorch_C_torch_normal_", (DL_FUNC) &_tinytorch_C_torch_normal_, 3},
    {"_tinytorch_C_torch_uniform_", (DL_FUNC) &_tinytorch_C_torch_uniform_, 3},
    {"_tinytorch_C_torch_zero_", (DL_FUNC) &_tinytorch_C_torch_zero_, 1},
    {"_tinytorch_C_torch_fill_", (DL_FUNC) &_tinytorch_C_torch_fill_, 2},
    {"_tinytorch_C_torch_repeat", (DL_FUNC) &_tinytorch_C_torch_repeat, 2},
    {"_tinytorch_C_torch_repeat_interleave", (DL_FUNC) &_tinytorch_C_torch_repeat_interleave, 3},
    {"_tinytorch_C_torch_index_select", (DL_FUNC) &_tinytorch_C_torch_index_select, 3},
    {"_tinytorch_C_torch_narrow", (DL_FUNC) &_tinytorch_C_torch_narrow, 4},
    {"_tinytorch_C_torch_scatter_", (DL_FUNC) &_tinytorch_C_torch_scatter_, 4},
    {"_tinytorch_C_tensor_to_device", (DL_FUNC) &_tinytorch_C_tensor_to_device, 2},
    {"_tinytorch_C_tensor_to_dtype_device", (DL_FUNC) &_tinytorch_C_tensor_to_dtype_device, 3},
    {"_tinytorch_C_torch_set_num_threads", (DL_FUNC) &_tinytorch_C_torch_set_num_threads, 1},
    {"_tinytorch_C_torch_get_num_threads", (DL_FUNC) &_tinytorch_C_torch_get_num_threads, 0},
    {"_tinytorch_C_torch_set_num_interop_threads", (DL_FUNC) &_tinytorch_C_torch_set_num_interop_threads, 1},
    {"_tinytorch_C_torch_get_num_interop_threads", (DL_FUNC) &_tinytorch_C_torch_get_num_interop_threads, 0},
    {"_tinytorch_C_cuda_is_available", (DL_FUNC) &_tinytorch_C_cuda_is_available, 0},
    {"_tinytorch_C_cuda_device_count", (DL_FUNC) &_tinytorch_C_cuda_device_count, 0},
    {"_tinytorch_C_cuda_empty_cache", (DL_FUNC) &_tinytorch_C_cuda_empty_cache, 0},
    {"_tinytorch_C_cuda_synchronize", (DL_FUNC) &_tinytorch_C_cuda_synchronize, 0},
    {"_tinytorch_C_cuda_mem_info", (DL_FUNC) &_tinytorch_C_cuda_mem_info, 0},
    {"_tinytorch_C_cuda_memory_stats", (DL_FUNC) &_tinytorch_C_cuda_memory_stats, 0},
    {"_tinytorch_C_torch_manual_seed", (DL_FUNC) &_tinytorch_C_torch_manual_seed, 1},
    {"_tinytorch_C_torch_scaled_mm", (DL_FUNC) &_tinytorch_C_torch_scaled_mm, 8},
    {"_tinytorch_C_transformer_decoder_layer_step", (DL_FUNC) &_tinytorch_C_transformer_decoder_layer_step, 7},
    {"_tinytorch_C_transformer_encoder_layer", (DL_FUNC) &_tinytorch_C_transformer_encoder_layer, 3},
    {"_tinytorch_C_prepare_cross_caches", (DL_FUNC) &_tinytorch_C_prepare_cross_caches, 3},
    {"_tinytorch_C_encoder_forward", (DL_FUNC) &_tinytorch_C_encoder_forward, 5},
    {"_tinytorch_C_decoder_forward_step", (DL_FUNC) &_tinytorch_C_decoder_forward_step, 9},
    {"_tinytorch_C_greedy_decode", (DL_FUNC) &_tinytorch_C_greedy_decode, 8},
    {NULL, NULL, 0}
};

extern "C" void R_init_tinytorch(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
