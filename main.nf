#!/usr/bin/env nextflow

nextflow.enable.dsl=2



/**
 * The make_input process generates synthetic input data
 * for an example run.
 */
process make_inputs {
    publishDir "${params.output_dir}"

    output:
        path("example.train.emx.txt"),      emit: train_data
        path("example.train.labels.txt"),   emit: train_labels
        path("example.perturb.emx.txt"),    emit: perturb_data
        path("example.perturb.labels.txt"), emit: perturb_labels
        path("example.genesets.txt"),       emit: gmt_file
        path("example.tsne.png")

    script:
        """
        make-inputs.py \
            --n-samples 1000 \
            --n-genes   100 \
            --n-classes 5 \
            --n-sets    2 \
            --tsne      example.tsne.png
        """
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
        path(train_data)
        path(train_labels)
        path(gmt_file)
        each gene_set

    output:
        tuple val(gene_set), path("target_model.h5"), emit: target_models
        path("train_target.log")

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
        each path(train_data)
        each path(train_labels)
        each path(gmt_file)
        tuple val(gene_set), path(target_model)
        each target

    output:
        tuple val(gene_set), val(target), path("*.h5"), emit: advgan_models
        path("*.train_advgan.log")

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
        each path(train_data)
        each path(train_labels)
        each path(perturb_data)
        each path(perturb_labels)
        each path(gmt_file)
        tuple val(gene_set), path(target_model), val(target), path(advgan_models)

    output:
        tuple val(gene_set), val(target), path("*.perturbations.samples.txt"), emit: sample_perturbations
        path("*.perturb.log")

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
        each path(train_data)
        each path(train_labels)
        each path(perturb_data)
        each path(perturb_labels)
        each path(gmt_file)
        tuple val(gene_set), val(target), path(sample_perturbations)

    output:
        path("*.png")
        path("*.visualize.log")

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



workflow {
    // create synthetic data if specified
    if ( params.make_inputs == true ) {
        make_inputs()
        train_data     = make_inputs.out.train_data
        train_labels   = make_inputs.out.train_labels
        perturb_data   = make_inputs.out.perturb_data
        perturb_labels = make_inputs.out.perturb_labels
        gmt_file       = make_inputs.out.gmt_file
    }

    // otherwise load input files
    else {
        train_data     = Channel.fromPath("${params.input_dir}/${params.train_data}")
        train_labels   = Channel.fromPath("${params.input_dir}/${params.train_labels}")
        perturb_data   = Channel.fromPath("${params.input_dir}/${params.perturb_data}")
        perturb_labels = Channel.fromPath("${params.input_dir}/${params.perturb_labels}")
        gmt_file       = Channel.fromPath("${params.input_dir}/${params.gmt_file}")
    }

    // extract gene set names from GMT file
    gene_sets = gmt_file.flatMap {
        it.readLines().collect { line -> line.tokenize("\t")[0] }
    }

    // train target model
    train_target(
        train_data,
        train_labels,
        gmt_file,
        gene_sets)

    target_models = train_target.out.target_models

    // parse target classes from params
    targets = Channel.fromList( params.targets.tokenize(',') )

    // train advgan model
    train_advgan(
        train_data,
        train_labels,
        gmt_file,
        target_models,
        targets)

    advgan_models = train_advgan.out.advgan_models

    // cross each target model with the corresponding
    // generator models for the perturb process
    models = target_models
        .cross(advgan_models)
        .map { it -> [it[0][0], it[0][1], it[1][1], it[1][2]] }

    // generate sample perturbations
    perturb(
        train_data,
        train_labels,
        perturb_data,
        perturb_labels,
        gmt_file,
        models)

    sample_perturbations = perturb.out.sample_perturbations

    // visualize t-sne and sample heatmaps
    visualize(
        train_data,
        train_labels,
        perturb_data,
        perturb_labels,
        gmt_file,
        sample_perturbations)
}
