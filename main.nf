#!/usr/bin/env nextflow



/**
 * Create channels for input files.
 */
if ( params.train_data != "" ) {
    TRAIN_DATA = Channel.fromPath("${params.input_dir}/${params.train_data}")
    TRAIN_LABELS = Channel.fromPath("${params.input_dir}/${params.train_labels}")
    PERTURB_DATA = Channel.fromPath("${params.input_dir}/${params.perturb_data}")
    PERTURB_LABELS = Channel.fromPath("${params.input_dir}/${params.perturb_labels}")
    GMT_FILE = Channel.fromPath("${params.input_dir}/${params.gmt_file}")
}
else {
    TRAIN_DATA = Channel.empty()
    TRAIN_LABELS = Channel.empty()
    PERTURB_DATA = Channel.empty()
    PERTURB_LABELS = Channel.empty()
    GMT_FILE = Channel.empty()
}



/**
 * The make_input process generates synthetic input data
 * for an example run.
 */
process make_inputs {
    publishDir "${params.output_dir}"

    output:
        file("example.train.emx.txt") into TRAIN_DATA_FROM_MAKE_INPUTS
        file("example.train.labels.txt") into TRAIN_LABELS_FROM_MAKE_INPUTS
        file("example.perturb.emx.txt") into PERTURB_DATA_FROM_MAKE_INPUTS
        file("example.perturb.labels.txt") into PERTURB_LABELS_FROM_MAKE_INPUTS
        file("example.genesets.txt") into GMT_FILE_FROM_MAKE_INPUTS
        file("example.tsne.png")

    when:
        params.make_inputs == true

    script:
        """
        make-inputs.py \
            --n-samples 1000 \
            --n-genes   100 \
            --n-classes 5 \
            --n-sets    5 \
            --tsne      example.tsne.png
        """
}



/**
 * Extract gene set names from each GMT file.
 */
GMT_FILE
    .mix(GMT_FILE_FROM_MAKE_INPUTS)
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
    .mix(TRAIN_DATA_FROM_MAKE_INPUTS)
    .into {
        TRAIN_DATA_FOR_TRAIN_TARGET;
        TRAIN_DATA_FOR_TRAIN_ADVGAN;
        TRAIN_DATA_FOR_PERTURB;
        TRAIN_DATA_FOR_VISUALIZE
    }

TRAIN_LABELS
    .mix(TRAIN_LABELS_FROM_MAKE_INPUTS)
    .into {
        TRAIN_LABELS_FOR_TRAIN_TARGET;
        TRAIN_LABELS_FOR_TRAIN_ADVGAN;
        TRAIN_LABELS_FOR_PERTURB;
        TRAIN_LABELS_FOR_VISUALIZE
    }

PERTURB_DATA
    .mix(PERTURB_DATA_FROM_MAKE_INPUTS)
    .into {
        PERTURB_DATA_FOR_PERTURB;
        PERTURB_DATA_FOR_VISUALIZE
    }

PERTURB_LABELS
    .mix(PERTURB_LABELS_FROM_MAKE_INPUTS)
    .into {
        PERTURB_LABELS_FOR_PERTURB;
        PERTURB_LABELS_FOR_VISUALIZE
    }



/**
 * The train_target process trains a target model, using a given
 * gene set as the input features.
 */
process train_target {
    publishDir pattern: "*.h5",  path: "${params.output_dir}/${gene_set}/models"
    publishDir pattern: "*.log", path: "${params.output_dir}/${gene_set}/logs"
    tag "${gene_set}"
    label "gpu"

    input:
        file(train_data) from TRAIN_DATA_FOR_TRAIN_TARGET
        file(train_labels) from TRAIN_LABELS_FOR_TRAIN_TARGET
        file(gmt_file) from GMT_FILE_FOR_TRAIN_TARGET
        each gene_set from GENE_SETS

    output:
        set val(gene_set), file("target_model.h5") into TARGET_MODELS
        file("train_target.log")

    script:
        """
        echo "#TRACE gene_set=${gene_set}"
        echo "#TRACE n_genes=`grep ${gene_set} ${gmt_file} | wc -w`"
        echo "#TRACE n_train_samples=`tail -n +1 ${train_data} | wc -l`"

        train-target.py \
            --dataset    ${train_data} \
            --labels     ${train_labels} \
            --gene-sets  ${gmt_file} \
            --set        ${gene_set} \
        > train_target.log
        """
}



/**
 * Send target models to each channel that consumes them.
 */
TARGET_MODELS
    .into {
        TARGET_MODELS_FOR_TRAIN_ADVGAN;
        TARGET_MODELS_FOR_PERTURB
    }



/**
 * The train_advgan process trains a generator model to perturb
 * samples to a given "target class", using a given set as the
 * input features.
 */
process train_advgan {
    publishDir pattern: "*.h5",  path: "${params.output_dir}/${gene_set}/models"
    publishDir pattern: "*.log", path: "${params.output_dir}/${gene_set}/logs"
    tag "${gene_set}/${target}"
    label "gpu"

    input:
        each file(train_data) from TRAIN_DATA_FOR_TRAIN_ADVGAN
        each file(train_labels) from TRAIN_LABELS_FOR_TRAIN_ADVGAN
        each file(gmt_file) from GMT_FILE_FOR_TRAIN_ADVGAN
        set val(gene_set), file(target_model) from TARGET_MODELS_FOR_TRAIN_ADVGAN
        each target from Channel.fromList( params.targets.tokenize(',') )

    output:
        set val(gene_set), val(target), file("*.h5") into ADVGAN_MODELS
        file("*.train_advgan.log")

    script:
        """
        echo "#TRACE gene_set=${gene_set}"
        echo "#TRACE n_genes=`grep ${gene_set} ${gmt_file} | wc -w`"
        echo "#TRACE n_train_samples=`tail -n +1 ${train_data} | wc -l`"

        train-advgan.py \
            --dataset    ${train_data} \
            --labels     ${train_labels} \
            --gene-sets  ${gmt_file} \
            --set        ${gene_set} \
            --target     ${target} \
            --target-cov ${params.target_cov} \
        > ${target}.train_advgan.log
        """
}



/**
 * Cross each target model with the corresponding
 * generator models for the perturb process.
 */
TARGET_MODELS_FOR_PERTURB
    .cross(ADVGAN_MODELS)
    .map { it -> [it[0][0], it[0][1], it[1][1], it[1][2]] }
    .set { MODELS_FOR_PERTURB }



/**
 * The perturb process uses a generator model, trained with a given
 * gene set and target class, to perturb samples from a "perturb" set
 * to the target class.
 */
process perturb {
    publishDir pattern: "*.txt", path: "${params.output_dir}/${gene_set}"
    publishDir pattern: "*.log", path: "${params.output_dir}/${gene_set}/logs"
    tag "${gene_set}/${target}"
    label "gpu"

    input:
        each file(train_data) from TRAIN_DATA_FOR_PERTURB
        each file(train_labels) from TRAIN_LABELS_FOR_PERTURB
        each file(perturb_data) from PERTURB_DATA_FOR_PERTURB
        each file(perturb_labels) from PERTURB_LABELS_FOR_PERTURB
        each file(gmt_file) from GMT_FILE_FOR_PERTURB
        set val(gene_set), file(target_model), val(target), file(advgan_models) from MODELS_FOR_PERTURB

    output:
        set val(gene_set), val(target), file("*.perturbations.samples.txt") into SAMPLE_PERTURBATIONS
        file("*.perturb.log")

    script:
        """
        echo "#TRACE gene_set=${gene_set}"
        echo "#TRACE n_genes=`grep ${gene_set} ${gmt_file} | wc -w`"
        echo "#TRACE n_train_samples=`tail -n +1 ${train_data} | wc -l`"
        echo "#TRACE n_perturb_samples=`tail -n +1 ${perturb_data} | wc -l`"

        perturb.py \
            --train-data     ${train_data} \
            --train-labels   ${train_labels} \
            --perturb-data   ${perturb_data} \
            --perturb-labels ${perturb_labels} \
            --gene-sets      ${gmt_file} \
            --set            ${gene_set} \
            --target         ${target} \
        > ${target}.perturb.log
        """
}



/**
 * The visualize process creates several visualizations of perturbed
 * samples for a given gene set and target class.
 */
process visualize {
    publishDir pattern: "*.*.*.png",  path: "${params.output_dir}/${gene_set}/heatmaps"
    publishDir pattern: "*.tsne.png", path: "${params.output_dir}/${gene_set}/tsne"
    publishDir pattern: "*.log",      path: "${params.output_dir}/${gene_set}/logs"
    tag "${gene_set}/${target}"

    input:
        each file(train_data) from TRAIN_DATA_FOR_VISUALIZE
        each file(train_labels) from TRAIN_LABELS_FOR_VISUALIZE
        each file(perturb_data) from PERTURB_DATA_FOR_VISUALIZE
        each file(perturb_labels) from PERTURB_LABELS_FOR_VISUALIZE
        each file(gmt_file) from GMT_FILE_FOR_VISUALIZE
        set val(gene_set), val(target), file(sample_perturbations) from SAMPLE_PERTURBATIONS

    output:
        file("*.png")
        file("*.visualize.log")

    script:
        """
        echo "#TRACE gene_set=${gene_set}"
        echo "#TRACE n_genes=`grep ${gene_set} ${gmt_file} | wc -w`"
        echo "#TRACE n_train_samples=`tail -n +1 ${train_data} | wc -l`"
        echo "#TRACE n_perturb_samples=`tail -n +1 ${perturb_data} | wc -l`"

        visualize.py \
            --train-data     ${train_data} \
            --train-labels   ${train_labels} \
            --perturb-data   ${perturb_data} \
            --perturb-labels ${perturb_labels} \
            --gene-sets      ${gmt_file} \
            --set            ${gene_set} \
            --target         ${target} \
            --tsne \
            --tsne-npca      ${params.tsne_npca} \
            --heatmap \
        > ${target}.visualize.log
        """
}
