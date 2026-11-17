function export_mdp(prefix, POut, RsOut, RhOut)
  for k=1:numel(POut)
    [r,c,v] = find(POut{k}); n = numel(r);
    fid = fopen(sprintf("%s_P%d.bin", prefix, k), "w"); fwrite(fid, n, "int32"); fwrite(fid, r, "int32"); fwrite(fid, c, "int32"); fwrite(fid, v, "double"); fclose(fid);
    [r,c,v] = find(RsOut{k}); n = numel(r);
    fid = fopen(sprintf("%s_Rs%d.bin", prefix, k), "w"); fwrite(fid, n, "int32"); fwrite(fid, r, "int32"); fwrite(fid, c, "int32"); fwrite(fid, v, "double"); fclose(fid);
    [r,c,v] = find(RhOut{k}); n = numel(r);
    fid = fopen(sprintf("%s_Rh%d.bin", prefix, k), "w"); fwrite(fid, n, "int32"); fwrite(fid, r, "int32"); fwrite(fid, c, "int32"); fwrite(fid, v, "double"); fclose(fid);
  end
  printf("exported %s nstates=%d\n", prefix, size(POut{1},1)); fflush(stdout);
end
