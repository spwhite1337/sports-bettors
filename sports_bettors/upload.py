parser = argparse.ArgumentParser(prog='Upload card classifier')
parser.add_argument('--windows', action='store_true')
parser.add_argument('--skipdata', action='store_true')
parser.add_argument('--skipresults', action='store_true')
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()

# General commands
sync_base = 'aws s3 sync '
dryrun_arg = ' --dryrun'
results_sync = '{} {}'.format(Config.CLOUD_RESULTS, Config.RESULTS_DIR)
data_sync = '{} {}'.format(Config.CLOUD_DATA, Config.DATA_DIR)

include_flag = " -- exclude '.gitignore'"
if args.windows:
    include_flag = re.sub("'", "", include_flag)

if not args.skipdata:
    logger.info('Uploading Data')
    sb_sync = sync_base + data_sync + include_flag
    sb_sync += dryrun_arg if args.dryrun else ''
    logger.info(sb_sync)
    os.system(sb_sync)
if not args.skipresults:
    logger.info('Uploading Results')
    sb_sync = sync_base + results_sync + include_flag
    sb_sync += dryrun_arg if args.dryun else ''
    logger.info(sb_sync)
    os.system(sb_sync)
