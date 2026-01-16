import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Upload, X, FileText, Image, FileSpreadsheet, FileCode } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file?: File;
}

interface FileUploadZoneProps {
  onFilesUploaded: (files: UploadedFile[]) => void;
}

export function FileUploadZone({ onFilesUploaded }: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      const uploadedFiles: UploadedFile[] = files.map((file) => ({
        id: Math.random().toString(36).substring(7),
        name: file.name,
        size: file.size,
        type: file.type,
        file: file,
      }));

      onFilesUploaded(uploadedFiles);
    },
    [onFilesUploaded]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      const uploadedFiles: UploadedFile[] = files.map((file) => ({
        id: Math.random().toString(36).substring(7),
        name: file.name,
        size: file.size,
        type: file.type,
        file: file,
      }));

      onFilesUploaded(uploadedFiles);
    },
    [onFilesUploaded]
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "border border-dashed rounded-none p-12 transition-all cursor-pointer relative overflow-hidden bg-secondary/40 shadow-inner hover:bg-secondary/60 transition-colors group/upload",
        isDragging
          ? "border-accent bg-accent/10"
          : "border-white/10"
      )}
    >
      <div className="paper-overlay opacity-[0.05]" />
      <input
        type="file"
        multiple
        onChange={handleFileSelect}
        className="hidden"
        id="file-upload"
        accept=".pdf,.doc,.docx,.xls,.xlsx,.png,.jpg,.jpeg,.txt"
      />
      <label htmlFor="file-upload" className="cursor-pointer relative z-10">
        <div className="flex flex-col items-center text-center">
          <motion.div
            animate={isDragging ? { y: -5 } : { y: 0 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Upload className="w-8 h-8 text-accent mb-6 opacity-60 group-hover/upload:opacity-100 transition-opacity" />
          </motion.div>
          <p className="text-sm font-serif font-bold uppercase tracking-widest text-primary mb-2">
            Ingest Archive
          </p>
          <p className="text-[11px] text-muted-foreground uppercase tracking-widest italic mb-6">
            Drag & Drop or Click to Research
          </p>
          <div className="flex gap-4 flex-wrap justify-center opacity-40 grayscale group-hover/upload:grayscale-0 transition-all duration-700">
            <span className="text-[9px] font-sans font-bold border-b border-accent/40 text-accent">PDF</span>
            <span className="text-[9px] font-sans font-bold border-b border-accent/40 text-accent">DOCX</span>
            <span className="text-[9px] font-sans font-bold border-b border-accent/40 text-accent">XLSX</span>
            <span className="text-[9px] font-sans font-bold border-b border-accent/40 text-accent">TXT</span>
          </div>
        </div>
      </label>
    </div>
  );
}

interface FileListProps {
  files: UploadedFile[];
  onRemove: (id: string) => void;
}

export function FileList({ files, onRemove }: FileListProps) {
  const getFileIcon = (type: string) => {
    if (type.includes("pdf")) return <FileText className="w-3.5 h-3.5 text-primary/60" />;
    if (type.includes("image")) return <Image className="w-3.5 h-3.5 text-primary/60" />;
    if (type.includes("spreadsheet") || type.includes("excel"))
      return <FileSpreadsheet className="w-3.5 h-3.5 text-primary/60" />;
    return <FileCode className="w-3.5 h-3.5 text-primary/60" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  if (files.length === 0) {
    return (
      <div className="text-center py-8 text-[10px] uppercase tracking-widest text-muted-foreground italic font-serif">
        Repository is currently empty
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {files.map((file) => (
        <motion.div
          key={file.id}
          initial={{ opacity: 0, x: -5 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 5 }}
          className="flex items-center justify-between p-4 border border-white/5 bg-secondary/40 backdrop-blur-md realistic-shadow group transition-all hover:bg-secondary/60"
        >
          <div className="flex items-center gap-4 flex-1 min-w-0">
            <div className="w-8 h-8 flex items-center justify-center bg-white/5 border border-white/10">
              {getFileIcon(file.type)}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-serif font-bold text-foreground truncate uppercase tracking-tight">{file.name}</p>
              <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
                {formatFileSize(file.size)} &bull; {file.type.split('/')[1] || 'DOC'}
              </p>
            </div>
          </div>
          <button
            onClick={() => onRemove(file.id)}
            className="p-2 hover:bg-accent/10 transition-colors group/btn"
          >
            <X className="w-3.5 h-3.5 text-muted-foreground group-hover/btn:text-accent transition-colors" />
          </button>
        </motion.div>
      ))}
    </div>
  );
}
