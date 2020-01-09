#!/usr/bin/env nextflow



/**
 * Create channels for input files.
 */
DATA_TXT_FILES = Channel.fromFilePairs("${params.input.dir}/${params.input.data_txt}", size: 1, flat: true)
DATA_NPY_FILES = Channel.fromFilePairs("${params.input.dir}/${params.input.data_npy}", size: 1, flat: true)
ROWNAME_FILES = Channel.fromFilePairs("${params.input.dir}/${params.input.rownames}", size: 1, flat: true)
COLNAME_FILES = Channel.fromFilePairs("${params.input.dir}/${params.input.colnames}", size: 1, flat: true)
LABEL_FILES = Channel.fromFilePairs("${params.input.dir}/${params.input.labels}", size: 1, flat: true)
GMT_FILES = Channel.fromFilePairs("${params.input.dir}/${params.input.gmt_files}", size: 1, flat: true)



/**
 * Group dataset files by dataset.
 */
LABEL_FILES.into {
	LABEL_FILES_FOR_TXT;
	LABEL_FILES_FOR_NPY
}

DATA_TXT_FILES
	.map { [it[0], [it[1]]] }
	.mix(LABEL_FILES_FOR_TXT)
	.groupTuple(size: 2)
	.map { [it[0], it[1].sort()] }
	.map { [it[0], it[1][0], it[1][1]] }
	.set { DATA_TXT_COMBINED }

DATA_NPY_FILES
	.mix(ROWNAME_FILES, COLNAME_FILES)
	.groupTuple()
	.map { [it[0], it[1].sort()] }
	.mix(LABEL_FILES_FOR_NPY)
	.groupTuple(size: 2)
	.map { [it[0], it[1].sort()] }
	.map { [it[0], it[1][1], it[1][0]] }
	.set { DATA_NPY_COMBINED }

Channel.empty()
	.mix(DATA_TXT_COMBINED, DATA_NPY_COMBINED)
	.set { DATASETS }



/**
 * Extract gene set names from each GMT file.
 */
GMT_FILES
	.flatMap { it[1].readLines().collect { line -> [it[0], it[1], line.tokenize("\t")[0]] } }
	.set { GENE_SETS }



/**
 * Combine datasets and gene sets, send combinations to each process.
 */
DATASETS
	.combine(GENE_SETS)
	.into {
		INPUTS_FOR_TRAIN_TARGET;
		INPUTS_FOR_TRAIN_ADVGAN;
		INPUTS_FOR_PERTURB;
		INPUTS_FOR_VISUALIZE
	}



/**
 * The train_target process trains a target model on a gene set.
 */
process train_target {
	tag "${dataset}/${gene_set}"
	publishDir "${params.output.dir}/${dataset}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

	input:
		set val(dataset), file(data_files), file(labels), val(gmt_name), file(gmt_file), val(gene_set) from INPUTS_FOR_TRAIN_TARGET

	output:
		set val(dataset), val(gene_set), file("target_model__*") into TARGET_MODELS_FROM_TRAIN_TARGET

	when:
		params.train_target.enabled == true

	script:
		"""
		train-target.py \
			--dataset    ${data_files[0]} \
			--labels     ${labels} \
			--gene-sets  ${gmt_file} \
			--set        ${gene_set} \
			--output-dir .

		rename 's/^target_model\\//target_model__/' target_model/*
		"""
}



TARGET_MODELS_FROM_TRAIN_TARGET
	.into {
		TARGET_MODELS_FOR_TRAIN_ADVGAN;
		TARGET_MODELS_FOR_PERTURB
	}



/**
 * The train_advgan process trains an AdvGAN model on a gene set.
 */
process train_advgan {
	tag "${dataset}/${gene_set}"
	publishDir "${params.output.dir}/${dataset}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

	input:
		set val(dataset), file(data_files), file(labels), val(gmt_name), file(gmt_file), val(gene_set) from INPUTS_FOR_TRAIN_ADVGAN
		set val(dataset), val(gene_set), file(target_model_files) from TARGET_MODELS_FOR_TRAIN_ADVGAN

	output:
		set val(dataset), val(gene_set), file("generator__*") into GENERATORS_FOR_PERTURB

	when:
		params.train_advgan.enabled == true

	script:
		"""
		mkdir -p target_model/

		rename 's/^target_model__/target_model\\//' target_model__*

		train-advgan.py \
			--dataset    ${data_files[0]} \
			--labels     ${labels} \
			--gene-sets  ${gmt_file} \
			--set        ${gene_set} \
			--target     ${params.input.target_class} \
			--output-dir .

		rename 's/^generator\\//generator__/' generator/*
		"""
}



/**
 * The perturb process generates perturbed samples using AdvGAN model.
 */
process perturb {
	tag "${dataset}/${gene_set}"
	publishDir "${params.output.dir}/${dataset}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

	input:
		set val(dataset), file(data_files), file(labels), val(gmt_name), file(gmt_file), val(gene_set) from INPUTS_FOR_PERTURB
		set val(dataset), val(gene_set), file(target_model_files) from TARGET_MODELS_FOR_PERTURB
		set val(dataset), val(gene_set), file(generator_files) from GENERATORS_FOR_PERTURB

	output:
		set val(dataset), val(gene_set), file("*.perturbed_means.txt") into PERTURBED_MEANS_FROM_PERTURB
		set val(dataset), val(gene_set), file("*.perturbed_samples.txt") into PERTURBED_SAMPLES_FROM_PERTURB

	when:
		params.perturb.enabled == true

	script:
		"""
		mkdir -p target_model/
		mkdir -p generator/

		rename 's/^target_model__/target_model\\//' target_model__*
		rename 's/^generator__/generator\\//' generator__*

		perturb.py \
			--train-data   ${data_files[0]} \
			--train-labels ${labels} \
			--test-data    ${data_files[0]} \
			--test-labels  ${labels} \
			--gene-sets    ${gmt_file} \
			--set          ${gene_set} \
			--target       ${params.input.target_class} \
			--output-dir   .
		"""
}



/**
 * The visualize process creates several visualizations of perturbed samples
 * for a gene set.
 */
process visualize {
	tag "${dataset}/${gene_set}"
	publishDir "${params.output.dir}/${dataset}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

	input:
		set val(dataset), file(data_files), file(labels), val(gmt_name), file(gmt_file), val(gene_set) from INPUTS_FOR_VISUALIZE
		set val(dataset), val(gene_set), file(perturbed_sample_files) from PERTURBED_SAMPLES_FROM_PERTURB

	output:
		file("*.png")

	when:
		params.visualize.enabled == true

	script:
		"""
		visualize.py \
			--train-data   ${data_files[0]} \
			--train-labels ${labels} \
			--test-data    ${data_files[0]} \
			--test-labels  ${labels} \
			--gene-sets    ${gmt_file} \
			--set          ${gene_set} \
			--target       ${params.input.target_class} \
			--output-dir   . \
			--tsne \
			--heatmap
		"""
}
