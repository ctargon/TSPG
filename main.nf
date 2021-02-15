#!/usr/bin/env nextflow



/**
 * Create channels for input files.
 */
TRAIN_DATA = Channel.fromPath("${params.input.dir}/${params.input.train_data}")
TRAIN_LABELS = Channel.fromPath("${params.input.dir}/${params.input.train_labels}")
PERTURB_DATA = Channel.fromPath("${params.input.dir}/${params.input.perturb_data}")
PERTURB_LABELS = Channel.fromPath("${params.input.dir}/${params.input.perturb_labels}")
GMT_FILE = Channel.fromPath("${params.input.dir}/${params.input.gmt_file}")



/**
 * Extract gene set names from each GMT file.
 */
GMT_FILE
    .into {
        GMT_FILE_FOR_GENE_SETS;
        GMT_FILE_FOR_TRAIN_TARGET;
        GMT_FILE_FOR_TRAIN_ADVGAN;
        GMT_FILE_FOR_PERTURB;
        GMT_FILE_FOR_VISUALIZE
    }

GMT_FILE_FOR_GENE_SETS
    .flatMap { it.readLines().collect { line -> line.tokenize("\t")[0] } }
    .set { GENE_SETS }



/**
 * Send inputs to each channel that consumes them.
 */
TRAIN_DATA
    .into {
        TRAIN_DATA_FOR_TRAIN_TARGET;
        TRAIN_DATA_FOR_TRAIN_ADVGAN;
        TRAIN_DATA_FOR_PERTURB;
        TRAIN_DATA_FOR_VISUALIZE
    }

TRAIN_LABELS
    .into {
        TRAIN_LABELS_FOR_TRAIN_TARGET;
        TRAIN_LABELS_FOR_TRAIN_ADVGAN;
        TRAIN_LABELS_FOR_PERTURB;
        TRAIN_LABELS_FOR_VISUALIZE
    }

PERTURB_DATA
    .into {
        PERTURB_DATA_FOR_PERTURB;
        PERTURB_DATA_FOR_VISUALIZE
    }

PERTURB_LABELS
    .into {
        PERTURB_LABELS_FOR_PERTURB;
        PERTURB_LABELS_FOR_VISUALIZE
    }



/**
 * The train_target process trains a target model on a gene set.
 */
process train_target {
    tag "${gene_set}"
    publishDir "${params.output.dir}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

    input:
        file(train_data) from TRAIN_DATA_FOR_TRAIN_TARGET
        file(train_labels) from TRAIN_LABELS_FOR_TRAIN_TARGET
        file(gmt_file) from GMT_FILE_FOR_TRAIN_TARGET
        each gene_set from GENE_SETS

    output:
        set val(gene_set), file("target_model__*") into TARGET_MODELS_FROM_TRAIN_TARGET

    script:
        """
        train-target.py \
            --dataset    ${train_data} \
            --labels     ${train_labels} \
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
    tag "${gene_set}"
    publishDir "${params.output.dir}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

    input:
        each file(train_data) from TRAIN_DATA_FOR_TRAIN_ADVGAN
        each file(train_labels) from TRAIN_LABELS_FOR_TRAIN_ADVGAN
        each file(gmt_file) from GMT_FILE_FOR_TRAIN_ADVGAN
        set val(gene_set), file(target_model_files) from TARGET_MODELS_FOR_TRAIN_ADVGAN

    output:
        set val(gene_set), file("generator__*") into GENERATORS_FOR_PERTURB

    script:
        """
        mkdir -p target_model/

        rename 's/^target_model__/target_model\\//' target_model__*

        find | sort

        train-advgan.py \
            --dataset    ${train_data} \
            --labels     ${train_labels} \
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
    tag "${gene_set}"
    publishDir "${params.output.dir}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

    input:
        each file(train_data) from TRAIN_DATA_FOR_PERTURB
        each file(train_labels) from TRAIN_LABELS_FOR_PERTURB
        each file(perturb_data) from PERTURB_DATA_FOR_PERTURB
        each file(perturb_labels) from PERTURB_LABELS_FOR_PERTURB
        each file(gmt_file) from GMT_FILE_FOR_PERTURB
        set val(gene_set), file(target_model_files) from TARGET_MODELS_FOR_PERTURB
        set val(gene_set), file(generator_files) from GENERATORS_FOR_PERTURB

    output:
        set val(gene_set), file("*.perturbations.means.txt") into MEAN_PERTURBATIONS
        set val(gene_set), file("*.perturbations.samples.txt") into SAMPLE_PERTURBATIONS

    script:
        """
        mkdir -p target_model/
        mkdir -p generator/

        rename 's/^target_model__/target_model\\//' target_model__*
        rename 's/^generator__/generator\\//' generator__*

        perturb.py \
            --train-data     ${train_data} \
            --train-labels   ${train_labels} \
            --perturb-data   ${perturb_data} \
            --perturb-labels ${perturb_labels} \
            --gene-sets      ${gmt_file} \
            --set            ${gene_set} \
            --target         ${params.input.target_class} \
            --output-dir     .
        """
}



/**
 * The visualize process creates several visualizations of perturbed samples
 * for a gene set.
 */
process visualize {
    tag "${gene_set}"
    publishDir "${params.output.dir}/${gene_set}", mode: "copy", saveAs: { it.replaceAll("__", "/") }

    input:
        each file(train_data) from TRAIN_DATA_FOR_VISUALIZE
        each file(train_labels) from TRAIN_LABELS_FOR_VISUALIZE
        each file(perturb_data) from PERTURB_DATA_FOR_VISUALIZE
        each file(perturb_labels) from PERTURB_LABELS_FOR_VISUALIZE
        each file(gmt_file) from GMT_FILE_FOR_VISUALIZE
        set val(gene_set), file(perturbed_sample_files) from SAMPLE_PERTURBATIONS

    output:
        file("*.png")

    script:
        """
        visualize.py \
            --train-data     ${train_data} \
            --train-labels   ${train_labels} \
            --perturb-data   ${perturb_data} \
            --perturb-labels ${perturb_labels} \
            --gene-sets      ${gmt_file} \
            --set            ${gene_set} \
            --target         ${params.input.target_class} \
            --output-dir     . \
            --tsne \
            --heatmap
        """
}
