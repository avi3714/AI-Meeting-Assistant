
import "./globals.css";

export const metadata = {
  title: "MeetMind | AI Meeting Assistant",
  description: "AI-powered meeting assistant",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
