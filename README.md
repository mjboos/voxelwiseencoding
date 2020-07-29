## Voxelwise Encoding BIDS App

This BIDS App lets you train voxelwise-encoding models for continuous (naturalistic) stimuli provided as a BIDS-compliant continuous recording file (see specification [here](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continuous-recordings.html)).

For more information about the specification of BIDS Apps see [here](https://docs.google.com/document/d/1E1Wi5ONvOVVnGhj21S1bmJJ4kyHFT7tkxnV3C23sjIE/).

### Description

TODO

### Documentation

TODO

### How to report errors

### Acknowledgments

### Usage

run.py /path/to/your/BIDS/dir /output/path --task your_task --ses session --skip_bids_validator --participant_label 01

This App has the following command line arguments:

		usage: run.py [-h]
		              [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
		              bids_dir output_dir {participant,group}

		Example BIDS App entry point script.

		positional arguments:
		  bids_dir              The directory with the input dataset formatted
		                        according to the BIDS standard.
		  output_dir            The directory where the output files should be stored.
		                        If you are running a group level analysis, this folder
		                        should be prepopulated with the results of
		                        the participant level analysis.

		optional arguments:
		  -h, --help            show this help message and exit
		  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
		                        The label(s) of the participant(s) that should be
		                        analyzed. The label corresponds to
		                        sub-<participant_label> from the BIDS spec (so it does
		                        not include "sub-"). If this parameter is not provided
		                        all subjects will be analyzed. Multiple participants
		                        can be specified with a space separated list.

To run it in participant level mode (for one participant):

    docker run -i --rm \
		-v /Users/filo/data/ds005:/bids_dataset:ro \
		-v /Users/filo/outputs:/outputs \
		bids/example \
		/bids_dataset /outputs participant --participant_label 01

After doing this for all subjects (potentially in parallel), the group level analysis
can be run:

    docker run -i --rm \
		-v /Users/filo/data/ds005:/bids_dataset:ro \
		-v /Users/filo/outputs:/outputs \
		bids/example \
		/bids_dataset /outputs group

### Special considerations
